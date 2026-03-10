"""
FetalAgent - Multi-agent fetal ultrasound analysis system

Usage:
    python main.py --inquiry "Estimate gestational age" --case_dir /path/to/case_dir

The case_dir should contain:
    - PNG/JPG images to analyze
    - pixel_size.csv (for GA/HC tasks): filename,pixel size(mm)
"""
import os
import re
import math
import asyncio
import shutil
import subprocess
import csv
import json
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image as PILImage

# AutoGen AgentChat
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import Image as AGImage

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

# Project root: directory containing this script
_SCRIPT_DIR = Path(__file__).resolve().parent
_CKPT_DIR = Path(os.environ.get("FETALAGENT_CKPT_DIR", str(_SCRIPT_DIR.parent / "FetalAgent_ckpt"))).resolve()


# =============================================
# Configuration
# =============================================
@dataclass
class ToolConfig:
    """Configuration for external tool paths.

    All paths default to locations relative to _SCRIPT_DIR.
    Override via environment variables or by modifying this dataclass.
    Conda environment Python paths must be set to match your installation.
    """

    # Conda environments -- set these to your own conda env paths
    fetal_base_python: str = os.environ.get("FETALAGENT_FETAL_BASE_PYTHON", "python")
    fetalclip_python: str = os.environ.get("FETALAGENT_FETALCLIP_PYTHON", "python")
    fetalclip2_python: str = os.environ.get("FETALAGENT_FETALCLIP2_PYTHON", "python")
    experiment_aaai_python: str = os.environ.get("FETALAGENT_EXPERIMENT_AAAI_PYTHON", "python")
    usfm_python: str = os.environ.get("FETALAGENT_USFM_PYTHON", "python")

    # Tool directories (relative to project root)
    aop_sam_dir: str = str(_SCRIPT_DIR / "external_tools" / "AoP_SAM")
    usfm_aop_dir: str = str(_SCRIPT_DIR / "external_tools" / "USFM_aop")
    csm_hc_dir: str = str(_SCRIPT_DIR / "external_tools" / "CSM_hc")
    usfm_hc_dir: str = str(_SCRIPT_DIR / "external_tools" / "USFM_hc")
    ga_algo1_dir: str = str(_SCRIPT_DIR / "external_tools" / "ga_radimagenet")
    ga_algo2_dir: str = str(_SCRIPT_DIR / "external_tools" / "ga_fetalclip")
    ga_algo3_dir: str = str(_SCRIPT_DIR / "external_tools" / "ga_convnext")
    keyframe_cls6_dir: str = str(_SCRIPT_DIR / "external_tools" / "keyframe_cls6")
    plane_fetalclip_dir: str = str(_SCRIPT_DIR / "external_tools" / "plane_fetalclip")
    plane_fulora_dir: str = str(_SCRIPT_DIR / "external_tools" / "plane_fulora")
    brain_subplane_fetalclip_dir: str = str(_SCRIPT_DIR / "external_tools" / "brain_subplane_fetalclip")
    agent_tools_dir: str = str(_SCRIPT_DIR / "tools")

    # Checkpoints -- expected in sibling folder FetalAgent_ckpt/
    aop_sam_ckpt: str = str(_CKPT_DIR / "aop_sam_fold0.pth")
    upernet_aop_ckpt: str = str(_CKPT_DIR / "upernet_aop_fold0.pth")
    brain_subplane_fetalclip_ckpt: str = str(_CKPT_DIR / "brain_subplane_fetalclip.ckpt")
    brain_subplane_resnet_ckpt: str = str(_CKPT_DIR / "brain_subplane_resnet.pth")
    brain_subplane_vit_ckpt: str = str(_CKPT_DIR / "brain_subplane_vit.pth")
    stomach_fetalclip_ckpt: str = str(_CKPT_DIR / "stomach_fetalclip.ckpt")
    stomach_samus_ckpt: str = str(_CKPT_DIR / "stomach_samus.pth")
    abdomen_fetalclip_ckpt: str = str(_CKPT_DIR / "abdomen_fetalclip.ckpt")
    abdomen_samus_ckpt: str = str(_CKPT_DIR / "abdomen_samus.pth")
    samus_base_ckpt: str = str(_CKPT_DIR / "SAMUS.pth")
    nnunet_predict: str = os.environ.get("FETALAGENT_NNUNET_PREDICT", "nnUNetv2_predict")
    keyframe_cls6_config: str = str(_SCRIPT_DIR / "external_tools" / "keyframe_cls6" / "config" / "classification.yml")

    # Timeouts
    default_timeout: int = 1800


# Global config instance
TOOL_CONFIG = ToolConfig()


# =============================================
# Model client configuration
# =============================================
def build_model_client() -> OpenAIChatCompletionClient:
    model_name = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAIChatCompletionClient(model=model_name, api_key=api_key, base_url=base_url)


# =============================================
# Image IO utilities
# =============================================
@dataclass
class ImageData:
    path: str
    array: Any
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    metadata: Dict[str, Any]


def _load_with_pillow(path: str) -> Optional[ImageData]:
    try:
        from PIL import Image
        img = Image.open(path)
        arr = np.array(img) if np is not None else img
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        meta = {"mode": getattr(img, "mode", None), "size": getattr(img, "size", None)}
        return ImageData(path=path, array=arr, spacing=spacing, origin=origin, direction=direction, metadata=meta)
    except Exception:
        return None


def load_image_any(path: str) -> ImageData:
    data = _load_with_pillow(path)
    if data is None:
        raise RuntimeError(f"Unable to load image: {path}")
    return data


# =============================================
# Tool Result Schema
# =============================================
@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    tool_name: str
    ok: bool
    per_image: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None
    logs: Dict[str, str] = field(default_factory=dict)
    artifacts_dir: Optional[str] = None


# =============================================
# Subprocess Runner
# =============================================
def run_tool_subprocess(
    python_path: str,
    script_path: str,
    args: List[str],
    cwd: Optional[str] = None,
    timeout: int = 1800,
    env_extra: Optional[Dict[str, str]] = None,
    log_prefix: Optional[str] = None,
    print_regexes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a tool script via subprocess, streaming output live to the console.
    Returns dict with returncode, combined stdout, and cmd.
    """
    import time
    import selectors

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    # Force unbuffered output so users can see progress while tools run
    env.setdefault("PYTHONUNBUFFERED", "1")

    prefix = log_prefix or os.path.basename(script_path)
    print_tool_output = os.environ.get("AGENT_PRINT_TOOL_OUTPUT", "1") not in ("0", "false", "False")
    print_tool_cmd = os.environ.get("AGENT_PRINT_TOOL_CMD", "0") in ("1", "true", "True", "yes", "Yes")
    heartbeat_sec = float(os.environ.get("AGENT_TOOL_HEARTBEAT_SEC", "60"))
    compiled_regexes: List[re.Pattern[str]] = []
    if print_regexes:
        compiled_regexes = [re.compile(p) for p in print_regexes]
    
    # -u: unbuffered
    cmd = [python_path, "-u", script_path] + args
    
    try:
        if print_tool_output and print_tool_cmd:
            print(f">>> [Tool:{prefix}] CMD: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,  # nosec - controlled internally
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)

        start = time.time()
        last_output = start
        out_chunks: List[str] = []

        while True:
            now = time.time()
            if timeout and (now - start) > timeout:
                try:
                    proc.kill()
                except Exception:
                    pass
                out_chunks.append(f"\n[{prefix}] ERROR: Timeout after {timeout}s\n")
                break

            events = sel.select(timeout=0.5)
            if events:
                for key, _mask in events:
                    line = key.fileobj.readline()
                    if not line:
                        # EOF
                        try:
                            sel.unregister(key.fileobj)
                        except Exception:
                            pass
                        break
                    out_chunks.append(line)
                    last_output = now
                    if print_tool_output:
                        # Print only result lines (and always print obvious errors)
                        is_errorish = ("Traceback" in line) or ("ERROR" in line) or ("Error" in line) or ("Exception" in line)
                        is_result = True
                        if compiled_regexes:
                            is_result = any(rx.search(line) for rx in compiled_regexes)
                        if is_errorish or is_result:
                            print(f"[{prefix}] {line}", end="", flush=True)
            else:
                # Heartbeat so users know we're still running even if the tool is silent
                if print_tool_output and heartbeat_sec > 0 and (now - last_output) >= heartbeat_sec:
                    print(f"[{prefix}] ...running...", flush=True)
                    last_output = now

            if proc.poll() is not None:
                # Drain remaining output
                try:
                    rest = proc.stdout.read()
                except Exception:
                    rest = ""
                if rest:
                    out_chunks.append(rest)
                    if print_tool_output:
                        for l in rest.splitlines(True):
                            is_errorish = ("Traceback" in l) or ("ERROR" in l) or ("Error" in l) or ("Exception" in l)
                            is_result = True
                            if compiled_regexes:
                                is_result = any(rx.search(l) for rx in compiled_regexes)
                            if is_errorish or is_result:
                                print(f"[{prefix}] {l}", end="", flush=True)
                break

        try:
            sel.close()
        except Exception:
            pass

        returncode = proc.returncode if proc.returncode is not None else -1
        stdout_text = "".join(out_chunks)
        return {
            "ok": returncode == 0,
            "returncode": returncode,
            "stdout": stdout_text,
            "stderr": "",  # merged into stdout
            "cmd": cmd,
        }

    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "cmd": cmd,
            "error": str(e),
        }


# =============================================
# Pixel Size CSV Utilities
# =============================================
def parse_pixel_size_csv(csv_path: str) -> Dict[str, float]:
    """Parse pixel_size.csv -> {filename: pixel_size_mm}."""
    pixel_sizes = {}
    if not os.path.exists(csv_path):
        return pixel_sizes
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                pixel_size_str = row.get("pixel size(mm)", "").strip()
                if filename and pixel_size_str:
                    try:
                        pixel_sizes[filename] = float(pixel_size_str)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[CSV] Error reading {csv_path}: {e}")
    return pixel_sizes


def ensure_pixel_csv(case_dir: str) -> str:
    """Ensure pixel_size.csv exists in case_dir. Return path."""
    csv_path = os.path.join(case_dir, "pixel_size.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    # Create default pixel_size.csv with 0.15 mm for all images
    images = [f for f in os.listdir(case_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filename,pixel size(mm)\n")
        for img in images:
            f.write(f"{img},0.15\n")
    print(f"[INFO] Created default pixel_size.csv with {len(images)} images")
    return csv_path


# =============================================
# GA ↔ HC cross-check utilities
# =============================================
# Parameters for log(HC) polynomial model, per percentile.
# HC(mm) = exp(b0 + b1*t + b2*t^2 + b3*t^3 + b4*t^4), where t is GA in weeks.
GA_TO_HC_PARAMS: Dict[str, List[float]] = {
    "0.025": [1.59317517131532, 2.9459800552433e-1, -7.3860372566707e-3, 6.56951770216148e-5, 0.0],
    "0.5":   [2.09924879247164, 2.53373656106037e-1, -6.05647816678282e-3, 5.14256072059917e-5, 0.0],
    "0.975": [2.50074069629423, 2.20067854715719e-1, -4.93623111462443e-3, 3.89066000946519e-5, 0.0],
}


def hc_from_ga_weeks(t_weeks: float, params: List[float]) -> float:
    """Compute HC(mm) from GA(weeks) using the provided polynomial-in-weeks model."""
    b0, b1, b2, b3, b4 = params
    return float(math.exp(b0 + b1 * t_weeks + b2 * (t_weeks ** 2) + b3 * (t_weeks ** 3) + b4 * (t_weeks ** 4)))


def hc_range_from_ga_weeks(t_weeks: float) -> Dict[str, float]:
    """Return HC(mm) at 2.5%, 50%, 97.5% for a given GA in weeks."""
    return {
        "p2_5": hc_from_ga_weeks(t_weeks, GA_TO_HC_PARAMS["0.025"]),
        "p50": hc_from_ga_weeks(t_weeks, GA_TO_HC_PARAMS["0.5"]),
        "p97_5": hc_from_ga_weeks(t_weeks, GA_TO_HC_PARAMS["0.975"]),
    }


def weeks_days_to_float_weeks(weeks: int, days: int) -> float:
    return float(weeks) + float(days) / 7.0


def float_weeks_to_weeks_days(t_weeks: float) -> Tuple[int, int]:
    w = int(t_weeks)
    d = int(round((t_weeks - w) * 7))
    if d >= 7:
        w += 1
        d -= 7
    if d < 0:
        d = 0
    return w, d


# =============================================
# AoP Tools
# =============================================
def run_aop_sam_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run AoP-SAM on images in case_dir."""
    script = os.path.join(config.agent_tools_dir, "aop_sam_step2_predict_agent.py")
    out_dir = _agent_outputs_dir("aop", "aop_sam_step2", case_dir)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = run_tool_subprocess(
        python_path=config.fetal_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--sam_ckpt",
            config.aop_sam_ckpt,
            "--out_dir",
            out_dir,
            "--gpu",
            gpu_id,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="AoP-SAM-step2",
        print_regexes=[r"\.png:\s*[\d.]+\s*deg"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse output: "filename.png: 123.45 deg | mask: /path/to/mask.png"
        for line in result["stdout"].splitlines():
            match = re.search(
                r"([^\s:]+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))\s*:\s*([-+]?\d*\.?\d+)\s*deg(?:\s*\|\s*mask:\s*(.+))?",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fname = match.group(1)
                aop = float(match.group(2))
                mask_path = (match.group(3) or "").strip() or None
                per_image[fname] = {"aop_deg": aop, "mask_path": mask_path}
    
    return ToolResult(
        tool_name="AoP-SAM",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_usfm_aop_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run USFM AoP on images in case_dir."""
    result = run_tool_subprocess(
        python_path=config.usfm_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir],
        cwd=config.usfm_aop_dir,
        timeout=config.default_timeout,
        log_prefix="USFM-AoP",
        print_regexes=[r"\.png:\s*[\d.]+\s*deg"],
    )
    
    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(
                r"([^\s:]+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))\s*:\s*([-+]?\d*\.?\d+)\s*deg(?:\s*\|\s*mask:\s*(.+))?",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fname = match.group(1)
                aop = float(match.group(2))
                mask_path = (match.group(3) or "").strip() or None
                per_image[fname] = {"aop_deg": aop, "mask_path": mask_path}
    
    return ToolResult(
        tool_name="USFM-AoP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_upernet_aop_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run UperNet AoP on images in case_dir."""
    script = os.path.join(config.agent_tools_dir, "upernet_aop_predict_agent.py")
    out_dir = _agent_outputs_dir("aop", "upernet", case_dir)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = run_tool_subprocess(
        python_path=config.fetal_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.upernet_aop_ckpt,
            "--out_dir",
            out_dir,
            "--gpu",
            gpu_id,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="UperNet-AoP",
        print_regexes=[r"\.png:\s*[\d.]+\s*deg"],
    )

    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(
                r"([^\s:]+\.(?:png|jpg|jpeg|bmp|tif|tiff|webp))\s*:\s*([-+]?\d*\.?\d+)\s*deg(?:\s*\|\s*mask:\s*(.+))?",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                fname = match.group(1)
                aop = float(match.group(2))
                mask_path = (match.group(3) or "").strip() or None
                per_image[fname] = {"aop_deg": aop, "mask_path": mask_path}

    return ToolResult(
        tool_name="UperNet-AoP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# =============================================
# HC Tools
# =============================================
def run_csm_hc_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run CSM HC measurement on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    out_dir = _agent_outputs_dir("head_circumference", "csm", case_dir)
    
    result = run_tool_subprocess(
        python_path=config.experiment_aaai_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir, "--pixel_csv", pixel_csv, "--output_dir", out_dir],
        cwd=config.csm_hc_dir,
        timeout=config.default_timeout,
        log_prefix="CSM-HC",
        # Suppress tool-emitted HC values (they use a different formula than our final pipeline).
        print_regexes=[r"^\bTHIS_REGEX_SHOULD_NOT_MATCH\b$"],
    )
    
    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)\s*mm", line)
            if match:
                fname = match.group(1)
                hc = float(match.group(2))
                mask_path = os.path.join(out_dir, "predictions", fname)
                per_image[fname] = {"hc_mm": hc, "mask_path": mask_path if os.path.exists(mask_path) else None}
    
    return ToolResult(
        tool_name="CSM-HC",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_usfm_hc_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run USFM HC on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    
    result = run_tool_subprocess(
        python_path=config.usfm_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.usfm_hc_dir,
        timeout=config.default_timeout,
        log_prefix="USFM-HC",
        print_regexes=[r"\.png:\s*[\d.]+\s*mm"],
    )
    
    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)\s*mm", line)
            if match:
                fname = match.group(1)
                hc = float(match.group(2))
                per_image[fname] = {"hc_mm": hc}
    
    return ToolResult(
        tool_name="USFM-HC",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def _largest_component_edge(mask: Any) -> Optional[Any]:
    if cv2 is None or np is None or mask is None or mask.size == 0:
        return None
    try:
        img = (mask > 0).astype("uint8") * 255
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=4)
        if retval <= 1:
            return None
        sort_label = np.argsort(-stats[:, 4])
        idx = labels == int(sort_label[1])
        max_connect = (idx * 255).astype("uint8")
        return cv2.Canny(max_connect, 50, 250)
    except Exception:
        return None


def _ellipse_circumference_mm_from_mask_array(mask: Optional[Any], pixel_size_mm: Optional[float]) -> Optional[float]:
    """
    Compute circumference (mm) from a binary mask by contour->fitEllipse->Ramanujan-II.
    Used for both HC and AC, aligned with eval scripts.
    """
    if cv2 is None or np is None or mask is None or pixel_size_mm is None:
        return None
    try:
        mask_bin = (mask > 0).astype("uint8")
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if contour is None or len(contour) < 5:
            return None
        (_, _), (major, minor), _ = cv2.fitEllipse(contour)
        a = max(float(major), float(minor)) / 2.0
        b = min(float(major), float(minor)) / 2.0
        if a <= 0 or b <= 0:
            return None
        circ_px = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
        return float(circ_px * float(pixel_size_mm))
    except Exception:
        return None


def _hc_mm_from_mask_array(mask: Optional[Any], pixel_size_mm: Optional[float]) -> Optional[float]:
    return _ellipse_circumference_mm_from_mask_array(mask, pixel_size_mm)


def _ac_mm_from_mask_array(mask: Optional[Any], pixel_size_mm: Optional[float]) -> Optional[float]:
    return _ellipse_circumference_mm_from_mask_array(mask, pixel_size_mm)


def _round_1dp(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 1)
    except Exception:
        return None


def _hadlock_ga_weeks_from_ac_mm(ac_mm: Optional[float]) -> Optional[float]:
    """
    Hadlock formula (AC-based):
      GA(weeks) = 8.14 + 0.0753 * AC(mm) + 0.000036 * AC(mm)^2
    """
    if ac_mm is None:
        return None
    try:
        ac = float(ac_mm)
        return 8.14 + 0.0753 * ac + 0.000036 * (ac ** 2)
    except Exception:
        return None


def _format_ga_weeks_days(t_weeks: Optional[float]) -> Optional[str]:
    if t_weeks is None:
        return None
    try:
        w, d = float_weeks_to_weeks_days(float(t_weeks))
        return f"{w}w {d}d"
    except Exception:
        return None


def _ga_label_to_weeks(ga_label: str) -> Optional[float]:
    if not ga_label:
        return None
    m = re.search(r"(\d+)\s*w\s*(\d+)\s*d", str(ga_label), flags=re.IGNORECASE)
    if not m:
        return None
    weeks = int(m.group(1))
    days = int(m.group(2))
    return weeks + days / 7.0


def _load_ga_reference_table(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ga_label = (row.get("GA(weeks/days)") or "").strip()
                ga_weeks = _ga_label_to_weeks(ga_label)
                if ga_weeks is None:
                    continue
                percentiles: Dict[float, float] = {}
                for k, v in row.items():
                    if k == "GA(weeks/days)":
                        continue
                    try:
                        pk = float(str(k).strip())
                        pv = float(str(v).strip())
                    except Exception:
                        continue
                    percentiles[pk] = pv
                if percentiles:
                    rows.append(
                        {
                            "ga_label": ga_label,
                            "ga_weeks": ga_weeks,
                            "percentiles": percentiles,
                        }
                    )
    except Exception as e:
        print(f"[Reference] Error reading {csv_path}: {e}")
        return []
    rows.sort(key=lambda x: float(x["ga_weeks"]))
    return rows


def _nearest_ga_row(table: List[Dict[str, Any]], ga_weeks: float) -> Optional[Dict[str, Any]]:
    if not table:
        return None
    return min(table, key=lambda r: abs(float(r["ga_weeks"]) - float(ga_weeks)))


def _fmt_percentile(p: float) -> str:
    if abs(p - round(p)) < 1e-9:
        return str(int(round(p)))
    return f"{p:.1f}".rstrip("0").rstrip(".")


def _percentile_assessment(
    measurement_mm: Optional[float],
    ga_weeks: Optional[float],
    table: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if measurement_mm is None or ga_weeks is None:
        return None
    row = _nearest_ga_row(table, ga_weeks)
    if not row:
        return None
    perc_map = row.get("percentiles") or {}
    if not isinstance(perc_map, dict) or not perc_map:
        return None
    sorted_p = sorted(float(p) for p in perc_map.keys())
    if not sorted_p:
        return None

    value = float(measurement_mm)
    p_low = 2.5 if 2.5 in perc_map else sorted_p[0]
    p_high = 97.5 if 97.5 in perc_map else sorted_p[-1]
    v_low = float(perc_map[p_low])
    v_high = float(perc_map[p_high])

    if value < v_low:
        status = "smaller"
    elif value > v_high:
        status = "larger"
    else:
        status = "within"

    # Locate percentile band from adjacent knots.
    band_lo = sorted_p[0]
    band_hi = sorted_p[-1]
    for i in range(len(sorted_p) - 1):
        p1 = sorted_p[i]
        p2 = sorted_p[i + 1]
        v1 = float(perc_map[p1])
        v2 = float(perc_map[p2])
        lo_v, hi_v = (v1, v2) if v1 <= v2 else (v2, v1)
        if lo_v <= value <= hi_v:
            band_lo = p1
            band_hi = p2
            break
        if value < float(perc_map[sorted_p[0]]):
            band_lo = sorted_p[0]
            band_hi = sorted_p[0]
        if value > float(perc_map[sorted_p[-1]]):
            band_lo = sorted_p[-1]
            band_hi = sorted_p[-1]

    return {
        "status": status,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_text": f"{_fmt_percentile(band_lo)}th-{_fmt_percentile(band_hi)}th Percentile",
        "normal_text": f"{_fmt_percentile(p_low)}th-{_fmt_percentile(p_high)}th Percentile",
        "ga_label_used": row.get("ga_label"),
    }


def _hc_percentile_sanity_check(
    recommended_hc_mm: Optional[float],
    alt_hc_mm: Optional[float],
    ga_weeks: Optional[float],
    hc_table: List[Dict[str, Any]],
    rec_source: str = "",
    alt_source: str = "",
) -> Tuple[Optional[float], str, str]:
    """Post-hoc HC plausibility check against GA-based percentile reference.

    Returns (final_hc_mm, final_source, note).
    If the recommended HC is outside 2.5-97.5 percentile for the given GA
    but the alternative tool's HC is within range, switch to the alternative.
    """
    if recommended_hc_mm is None or ga_weeks is None:
        return recommended_hc_mm, rec_source, "no_check"
    rec_assess = _percentile_assessment(recommended_hc_mm, ga_weeks, hc_table)
    if rec_assess is None:
        return recommended_hc_mm, rec_source, "no_reference_data"
    if rec_assess["status"] == "within":
        return recommended_hc_mm, rec_source, "in_range"
    if alt_hc_mm is not None:
        alt_assess = _percentile_assessment(alt_hc_mm, ga_weeks, hc_table)
        if alt_assess is not None and alt_assess["status"] == "within":
            return (
                _round_1dp(alt_hc_mm),
                alt_source,
                f"switched: {rec_source} HC {recommended_hc_mm:.1f} mm out of range "
                f"({rec_assess['status']}), {alt_source} HC {alt_hc_mm:.1f} mm is within range",
            )
    return recommended_hc_mm, rec_source, f"kept: both tools out of range ({rec_assess['status']})"


def _extract_lmp_ga_weeks(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.lower()
    # "GA (LMP) is 24.5"
    m = re.search(r"ga\s*\(\s*lmp\s*\)\s*is\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        return float(m.group(1))
    # "LMP ... 24w 3d" / "last menstrual period ... 24 weeks 3 days"
    m = re.search(
        r"(?:lmp|last menstrual period)[^0-9]{0,30}(\d+)\s*w(?:eeks?)?(?:[^0-9]{0,10}(\d+)\s*d(?:ays?)?)?",
        t,
    )
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) else 0
        return w + d / 7.0
    return None


def _plane_display_name(plane: Optional[str]) -> str:
    p = (plane or "").strip().lower()
    if p == "brain":
        return "Fetal Brain"
    if p == "abdomen":
        return "Fetal Abdomen"
    if p == "thorax":
        return "Fetal Thorax"
    if p == "femur":
        return "Fetal Femur"
    return "Unknown"


def _parse_expert_per_image(expert_outputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    for item in expert_outputs:
        task = str(item.get("task") or "")
        txt = item.get("expert_text") or ""
        try:
            data = json.loads(txt)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("per_image"), dict):
            parsed[task] = data["per_image"]
    return parsed


def _build_structured_text_summary(
    user_inquiry: str,
    images: List[str],
    expert_outputs: List[Dict[str, Any]],
) -> str:
    parsed = _parse_expert_per_image(expert_outputs)
    plane_map = parsed.get("plane_classification", {})
    brain_map = parsed.get("brain_subplanes", {})
    hc_map = parsed.get("head_circumference", {})
    ga_map = parsed.get("gestational_age", {})
    abd_map = parsed.get("abdomen_segmentation", {})
    sto_map = parsed.get("stomach_segmentation", {})

    # Extract HC algo_results for the percentile sanity check.
    hc_algo_final: Dict[str, Dict[str, Any]] = {}
    for item in expert_outputs:
        if item.get("task") == "head_circumference":
            hc_algo_final = (item.get("algo_results") or {}).get("final_hc") or {}
            break

    hc_ref_path = str(_SCRIPT_DIR / "reference" / "HC_GA_reference.csv")
    ac_ref_path = str(_SCRIPT_DIR / "reference" / "AC_GA_reference.csv")
    hc_table = _load_ga_reference_table(hc_ref_path)
    ac_table = _load_ga_reference_table(ac_ref_path)
    lmp_ga_weeks = _extract_lmp_ga_weeks(user_inquiry)

    image_names = set(images)
    for task_map in (plane_map, brain_map, hc_map, ga_map, abd_map, sto_map):
        image_names.update(task_map.keys())

    reports: List[str] = []
    for fname in sorted(image_names):
        plane = (plane_map.get(fname) or {}).get("recommended")
        brain_plane = (brain_map.get(fname) or {}).get("recommended")
        hc_mm = (hc_map.get(fname) or {}).get("recommended")
        hc_mask = (hc_map.get(fname) or {}).get("recommended_mask_path")
        ga_rec = (ga_map.get(fname) or {}).get("recommended") or {}
        ac_mm = (abd_map.get(fname) or {}).get("recommended_ac_mm")
        ac_ga = (abd_map.get(fname) or {}).get("recommended_ga_weeks_from_ac")
        abd_mask = (abd_map.get(fname) or {}).get("recommended_mask_path")
        sto_mask = (sto_map.get(fname) or {}).get("recommended")

        ga_us_weeks: Optional[float] = None
        if isinstance(ga_rec, dict):
            wk = ga_rec.get("weeks")
            dy = ga_rec.get("days")
            if wk is not None and dy is not None:
                try:
                    ga_us_weeks = float(wk) + float(dy) / 7.0
                except Exception:
                    ga_us_weeks = None

        # Post-hoc HC sanity check against percentile reference.
        if hc_mm is not None and ga_us_weeks is not None and hc_algo_final:
            hc_detail = hc_algo_final.get(fname, {})
            rec_src = hc_detail.get("source", "")
            csm_val = hc_detail.get("csm_hc_mm")
            nn_val = hc_detail.get("nnunet_hc_mm")
            if "csm" in rec_src:
                alt_val, alt_src = nn_val, "nnunet"
            else:
                alt_val, alt_src = csm_val, "csm"
            checked_hc, checked_src, check_note = _hc_percentile_sanity_check(
                float(hc_mm), alt_val, ga_us_weeks,
                hc_table, rec_src, alt_src,
            )
            if checked_hc is not None and abs(float(checked_hc) - float(hc_mm)) > 0.05:
                print(f"    [HC sanity check] {check_note}")
                hc_mm = checked_hc
                if "csm" in checked_src:
                    hc_mask = hc_detail.get("csm_mask_path") or hc_mask
                else:
                    hc_mask = hc_detail.get("nnunet_mask_path") or hc_mask

        findings: List[str] = []
        findings.append("Findings:")
        findings.append(f"Plane Identification: {_plane_display_name(plane)}")
        if brain_plane and str(brain_plane).upper() != "N/A":
            findings.append(f"Brain Plane Classification: {brain_plane}")
        if hc_mask:
            findings.append(f"Fetal Brain segmentation mask created in {hc_mask}")
        if abd_mask:
            findings.append(f"Abdomen segmentation mask created in {abd_mask}")
        if sto_mask:
            findings.append(f"Stomach segmentation mask created in {sto_mask}")
        if hc_mm is not None:
            findings.append(f"Head Circumference (HC) {float(hc_mm):.1f} mm.")
        if ac_mm is not None:
            findings.append(f"Estimated Abdomen Circumference (AC) {float(ac_mm):.1f} mm.")
        ga_us_text = _format_ga_weeks_days(ga_us_weeks)
        ga_ac_text = _format_ga_weeks_days(float(ac_ga) if ac_ga is not None else None)
        lmp_text = _format_ga_weeks_days(lmp_ga_weeks)
        if ga_us_text is not None:
            findings.append(f"Estimated Gestational Age (GA) {ga_us_text}.")
        if ga_ac_text is not None:
            findings.append(f"Estimated Gestational Age (GA) {ga_ac_text} (from Hadlock formula).")

        impression: List[str] = []
        impression.append("")
        impression.append("Impression:")
        ga_for_impression = ga_us_weeks if ga_us_weeks is not None else ac_ga
        ga_imp_text = _format_ga_weeks_days(float(ga_for_impression) if ga_for_impression is not None else None)
        if ga_imp_text is not None:
            impression.append(f"Estimated fetal age {ga_imp_text} by ultrasound measurement.")
        if lmp_text is not None:
            impression.append(f"Estimated fetal age {lmp_text} from last menstrual period (LMP).")

        # Growth statement priority:
        # 1) If LMP GA is present and AC is available -> AC vs LMP reference.
        # 2) Else if HC and GA(US) are available -> HC vs GA(US) reference.
        growth_line = None
        if lmp_ga_weeks is not None and ac_mm is not None:
            ac_assess = _percentile_assessment(ac_mm, lmp_ga_weeks, ac_table)
            if ac_assess:
                if ac_assess["status"] == "within":
                    growth_line = (
                        f"Compared with GA estimated from last menstrual period (LMP), AC falls within normal fetal growth range "
                        f"({ac_assess['band_text']})."
                    )
                elif ac_assess["status"] == "larger":
                    growth_line = (
                        f"Compared with GA estimated from last menstrual period (LMP), AC is larger than normal fetal growth range "
                        f"(>{ac_assess['normal_text'].split('-')[-1]})."
                    )
                else:
                    growth_line = (
                        f"Compared with GA estimated from last menstrual period (LMP), AC is smaller than normal fetal growth range "
                        f"(<{ac_assess['normal_text'].split('-')[0]})."
                    )
        elif hc_mm is not None and ga_us_weeks is not None:
            hc_assess = _percentile_assessment(hc_mm, ga_us_weeks, hc_table)
            if hc_assess:
                if hc_assess["status"] == "within":
                    growth_line = f"HC falls within normal fetal growth range ({hc_assess['band_text']})."
                elif hc_assess["status"] == "larger":
                    growth_line = f"HC is larger than normal fetal growth range (>{hc_assess['normal_text'].split('-')[-1]})."
                else:
                    growth_line = f"HC is smaller than normal fetal growth range (<{hc_assess['normal_text'].split('-')[0]})."
        if growth_line:
            impression.append(growth_line)

        report_text = "\n".join([f"Image: {fname}", ""] + findings + impression)
        reports.append(report_text)

    return "\n\n" + ("\n\n" + ("-" * 60) + "\n\n").join(reports)


def run_nnunet_hc_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run nnUNet HC segmentation and derive HC(mm) from predicted mask."""
    script = os.path.join(config.agent_tools_dir, "nnunet_hc_seg_predict_agent.py")
    out_dir = _agent_outputs_dir("head_circumference", "nnunet", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetal_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--out_dir",
            out_dir,
            "--nnunet_predict",
            config.nnunet_predict,
            "--timeout",
            str(config.default_timeout),
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout + 120,
        log_prefix="HC-nnUNet",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        mask_paths = _parse_filename_colon_value(result["stdout"])
        pixel_map = parse_pixel_size_csv(os.path.join(case_dir, "pixel_size.csv"))
        for fname, mask_path in mask_paths.items():
            raw_img = _safe_load_pil(os.path.join(case_dir, fname))
            mask_arr = _mask_to_raw_array(mask_path, raw_img, preprocess="resize_direct")
            hc_mm = _hc_mm_from_mask_array(mask_arr, pixel_map.get(fname))
            per_image[fname] = {"mask_path": mask_path, "hc_mm": hc_mm}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="HC-nnUNet",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# =============================================
# GA Tools
# =============================================
def run_ga_algo1_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run GA Algorithm 1 (RadImageNet-based) on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    
    result = run_tool_subprocess(
        python_path=config.fetal_base_python,
        script_path="predict_agent.py",
        args=["--img_dir", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.ga_algo1_dir,
        timeout=config.default_timeout,
        log_prefix="GA-RadImageNet",
        print_regexes=[r"\.png:\s*[-+]?(?:\d+\.?\d*|\.\d+)\s*$"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse: "filename.png: 25.1234"
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)", line)
            if match:
                fname = match.group(1)
                ga_weeks = float(match.group(2))
                weeks = int(ga_weeks)
                days = int((ga_weeks - weeks) * 7)
                per_image[fname] = {"ga_weeks": weeks, "ga_days": days, "total_weeks": ga_weeks}
    
    return ToolResult(
        tool_name="GA-RadImageNet",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_ga_algo2_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run GA Algorithm 2 (FetalCLIP-based) on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)
    
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path="predict_agent.py",
        args=["--img_dir", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.ga_algo2_dir,
        timeout=config.default_timeout,
        log_prefix="GA-FetalCLIP",
        print_regexes=[r"^\[[^\]]+\.png\]\s+Predicted GA"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse: "[filename.png] Predicted GA ≈ 25 weeks + 3 days (25.4286 days total)"
        for line in result["stdout"].splitlines():
            match = re.search(
                r"\[([^\]]+\.png)\]\s+Predicted GA[^0-9]*(\d+)\s+weeks?\s*\+\s*(\d+)\s+days?.*\(([\d.]+)",
                line
            )
            if match:
                fname = match.group(1)
                weeks = int(match.group(2))
                days = int(match.group(3))
                total = float(match.group(4))
                per_image[fname] = {"ga_weeks": weeks, "ga_days": days, "total_weeks": total}
    
    return ToolResult(
        tool_name="GA-FetalCLIP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_ga_algo3_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run GA Algorithm 3 (ConvNeXt-based) on images in case_dir."""
    pixel_csv = ensure_pixel_csv(case_dir)

    result = run_tool_subprocess(
        python_path=config.fetal_base_python,
        script_path="predict_agent.py",
        args=["--img_dir", case_dir, "--pixel_csv", pixel_csv],
        cwd=config.ga_algo3_dir,
        timeout=config.default_timeout,
        log_prefix="GA-ConvNeXt",
        print_regexes=[r"\.png:\s*[-+]?(?:\d+\.?\d*|\.\d+)\s*$"],
    )

    per_image = {}
    if result["ok"]:
        for line in result["stdout"].splitlines():
            match = re.search(r"([^\s:]+\.png)\s*:\s*([\d.]+)", line)
            if match:
                fname = match.group(1)
                ga_weeks = float(match.group(2))
                weeks = int(ga_weeks)
                days = int((ga_weeks - weeks) * 7)
                per_image[fname] = {"ga_weeks": weeks, "ga_days": days, "total_weeks": ga_weeks}

    return ToolResult(
        tool_name="GA-ConvNeXt",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# =============================================
# Plane Classification Tools
# =============================================
def run_plane_fetalclip_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run FetalCLIP plane classification on images in case_dir."""
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir],
        cwd=config.plane_fetalclip_dir,
        timeout=config.default_timeout,
        log_prefix="Plane-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*→\s*Predicted plane:"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse:
        #   filename.png → Predicted plane: brain | probs: abdomen:0.1,brain:0.8,...
        # or fallback without probs.
        for line in result["stdout"].splitlines():
            match = re.search(rf"([^\s→]+\.{_FNAME_EXT_RE})\s*→\s*Predicted plane:\s*([^|]+)\|\s*probs:\s*(.+)", line, flags=re.IGNORECASE)
            if match:
                fname = match.group(1).strip()
                plane = match.group(2).strip()
                prob_str = match.group(3).strip()
                probs: Dict[str, float] = {}
                for kv in prob_str.split(","):
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        try:
                            probs[k.strip()] = float(v.strip())
                        except Exception:
                            pass
                per_image[fname] = {"plane": plane, "probs": probs}
                continue
            match2 = re.search(rf"([^\s→]+\.{_FNAME_EXT_RE})\s*→\s*Predicted plane:\s*(.+)", line, flags=re.IGNORECASE)
            if match2:
                fname = match2.group(1).strip()
                plane = match2.group(2).strip()
                per_image[fname] = {"plane": plane, "probs": {}}
    
    return ToolResult(
        tool_name="Plane-FetalCLIP",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_plane_fulora_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run FU-LoRA plane classification on images in case_dir."""
    result = run_tool_subprocess(
        python_path=config.experiment_aaai_python,
        script_path="predict_agent.py",
        args=["--data_path", case_dir],
        cwd=config.plane_fulora_dir,
        timeout=config.default_timeout,
        log_prefix="Plane-FU-LoRA",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )
    
    per_image = {}
    if result["ok"]:
        # Parse:
        #   filename.png: Fetal brain | probs: Other:0.01,Fetal abdomen:0.02,...
        # or fallback without probs.
        for line in result["stdout"].splitlines():
            match = re.search(rf"([^\s:]+\.{_FNAME_EXT_RE})\s*:\s*([^|]+)\|\s*probs:\s*(.+)", line, flags=re.IGNORECASE)
            if match:
                fname = match.group(1).strip()
                plane = match.group(2).strip()
                prob_str = match.group(3).strip()
                probs: Dict[str, float] = {}
                for kv in prob_str.split(","):
                    if ":" in kv:
                        parts = kv.rsplit(":", 1)
                        if len(parts) == 2:
                            k, v = parts
                            try:
                                probs[k.strip()] = float(v.strip())
                            except Exception:
                                pass
                per_image[fname] = {"plane": plane, "probs": probs}
                continue
            match2 = re.search(rf"([^\s:]+\.{_FNAME_EXT_RE})\s*:\s*(.+)", line, flags=re.IGNORECASE)
            if match2:
                fname = match2.group(1).strip()
                plane = match2.group(2).strip()
                per_image[fname] = {"plane": plane, "probs": {}}
    
    return ToolResult(
        tool_name="Plane-FU-LoRA",
        ok=result["ok"] and len(per_image) > 0,
        per_image=per_image,
        error=result.get("error") or (None if result["ok"] else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# =============================================
# New Tools: Brain subplane / Stomach seg / Abdomen seg
# =============================================
_FNAME_EXT_RE = r"(?:png|jpg|jpeg|bmp|tif|tiff|webp)"


def _agent_outputs_dir(task: str, tool: str, case_dir: str) -> str:
    """Writable output folder under FetalAgent_hjw/outputs_agent/."""
    base = Path(__file__).resolve().parent
    out = base / "outputs_agent" / task / tool / Path(case_dir).name
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _safe_load_pil(path: str) -> Optional[PILImage.Image]:
    try:
        return PILImage.open(path).convert("RGB")
    except Exception:
        return None


def _make_square_pil(image: PILImage.Image) -> PILImage.Image:
    """Pad to square with black background (matches tool-side preprocessing)."""
    width, height = image.size
    max_side = max(width, height)
    new_image = PILImage.new("RGB", (max_side, max_side), (0, 0, 0))
    padding_left = (max_side - width) // 2
    padding_top = (max_side - height) // 2
    new_image.paste(image.convert("RGB"), (padding_left, padding_top))
    return new_image


# def _make_overlay(raw_img: PILImage.Image, mask_path: Optional[str], color=(0, 255, 0, 120)) -> Optional[PILImage.Image]:
#     """Create a semi-transparent overlay of mask on raw image."""
#     if not mask_path or (mask_path and not os.path.exists(mask_path)):
#         return None
#     try:
#         mask_img = PILImage.open(mask_path)
#     except Exception:
#         return None

#     # Important: these tools often predict masks at fixed resolution (e.g., 224x224 or 256x256)
#     # after a square-pad+resize preprocessing. To keep overlays aligned (and avoid PIL size errors),
#     # we create the overlay on a square-padded raw image resized to the mask resolution.
#     mask_l = mask_img.convert("L")
#     base = raw_img
#     if base.size != mask_l.size:
#         base = _make_square_pil(base)
#         base = base.resize(mask_l.size, resample=PILImage.BICUBIC)
#     raw_rgba = base.convert("RGBA")

#     if mask_l.size != raw_rgba.size:
#         mask_l = mask_l.resize(raw_rgba.size, resample=PILImage.NEAREST)

#     # Build alpha channel where mask > 0
#     alpha = mask_l.point(lambda p: color[3] if p > 0 else 0)
#     color_layer = PILImage.new("RGBA", raw_rgba.size, (color[0], color[1], color[2], 0))
#     color_layer.putalpha(alpha)
#     return PILImage.alpha_composite(raw_rgba, color_layer)
def _make_overlay(
    raw_img: PILImage.Image,
    mask_path: Optional[str],
    preprocess: str = "resize_direct",  # "resize_direct" | "pad_square"
    color=(0, 255, 0, 120),
) -> Optional[PILImage.Image]:
    """
    Overlay a predicted mask onto the ORIGINAL raw image coordinate system.

    preprocess:
      - "resize_direct": tool resized raw (H,W) -> (S,S) without padding
      - "pad_square": tool padded raw to square then resized to (S,S)
    """
    if not mask_path or not os.path.exists(mask_path) or raw_img is None:
        return None

    try:
        mask = PILImage.open(mask_path).convert("L")
    except Exception:
        return None

    w, h = raw_img.size

    if preprocess == "resize_direct":
        # Inverse of direct resize: resize mask back to raw size
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)

    elif preprocess == "pad_square":
        # Inverse of pad-to-square + resize:
        # 1) bring mask to square size (max_side, max_side)
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2

        mask_sq = mask.resize((max_side, max_side), resample=PILImage.NEAREST)

        # 2) crop out the original image region (remove padding)
        mask_on_raw = mask_sq.crop((pad_left, pad_top, pad_left + w, pad_top + h))

    else:
        # Fallback: do something safe
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)

    raw_rgba = raw_img.convert("RGBA")

    alpha = mask_on_raw.point(lambda p: color[3] if p > 0 else 0)
    color_layer = PILImage.new("RGBA", raw_rgba.size, (color[0], color[1], color[2], 0))
    color_layer.putalpha(alpha)

    return PILImage.alpha_composite(raw_rgba, color_layer)



def _concat_side_by_side(images: List[PILImage.Image]) -> Optional[PILImage.Image]:
    imgs = [im for im in images if im is not None]
    if not imgs:
        return None
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    canvas = PILImage.new("RGB", (total_width, max_height), (0, 0, 0))
    x = 0
    for im in imgs:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.size[0]
    return canvas


def _pil_to_agimage(img: PILImage.Image) -> AGImage:
    return AGImage(img)


def _parse_filename_colon_value(stdout: str) -> Dict[str, str]:
    pat = re.compile(rf"^(?P<fname>.+\.{_FNAME_EXT_RE})\s*:\s*(?P<val>.+)\s*$", flags=re.IGNORECASE)
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if not m:
                continue
        out[m.group("fname").strip()] = m.group("val").strip()
    return out


def _parse_filename_label_probs(stdout: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse lines in format:
      filename.png: Label [0.12 0.34 0.54]
      filename.png: Label
    """
    pat = re.compile(
        rf"(?P<fname>[^\s:]+\.(?:{_FNAME_EXT_RE}))\s*:\s*(?P<label>[^\[]+?)(?:\s*\[(?P<probs>[^\]]+)\])?\s*$",
        flags=re.IGNORECASE,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for line in stdout.splitlines():
        m = pat.search(line.strip())
        if not m:
            continue
        fname = m.group("fname").strip()
        label = m.group("label").strip()
        probs_raw = m.group("probs")
        probs: Optional[List[float]] = None
        if probs_raw:
            try:
                probs = [float(x) for x in probs_raw.strip().split()]
            except Exception:
                probs = None
        out[fname] = {"label": label, "probs": probs}
    return out


def _parse_filename_colon_text(stdout: str) -> Dict[str, str]:
    pat = re.compile(rf"^(?P<fname>.+\.{_FNAME_EXT_RE})\s*:\s*(?P<val>.+?)\s*$", flags=re.IGNORECASE)
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        out[m.group("fname").strip()] = m.group("val").strip()
    return out


def _normalize_video_plane_label(label: Optional[str]) -> str:
    s = (label or "").strip().lower()
    if "biparietal" in s or "brain" in s:
        return "brain"
    if "abdominal" in s or "abdomen" in s:
        return "abdomen"
    if "femur" in s:
        return "femur"
    if "heart" in s or "thorax" in s:
        return "thorax"
    if "spine" in s:
        return "spine"
    if "no plane" in s or s == "no_plane":
        return "other"
    return "other"


def _is_video_summary_request(inquiry: str) -> bool:
    t = (inquiry or "").lower()
    has_video = any(k in t for k in ("video", "continuous screenshot", "continuous screenshots", "cine", "sequence"))
    has_summary = any(k in t for k in ("summary", "comprehensive", "caption"))
    return has_video and has_summary


def _resolve_image_key(pred_name: str, available_images: List[str]) -> Optional[str]:
    pred_stem = Path(pred_name).stem.lower()
    pred_full = pred_name.lower()
    for x in available_images:
        if x.lower() == pred_full:
            return x
    for x in available_images:
        if Path(x).stem.lower() == pred_stem:
            return x
    return None


def _make_single_image_case_dir(case_dir: str, image_name: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="agent_single_case_")
    src_img = os.path.join(case_dir, image_name)
    dst_img = os.path.join(tmp_dir, image_name)
    shutil.copy2(src_img, dst_img)

    src_csv = os.path.join(case_dir, "pixel_size.csv")
    if os.path.exists(src_csv):
        pixel_map = parse_pixel_size_csv(src_csv)
        px = pixel_map.get(image_name)
        with open(os.path.join(tmp_dir, "pixel_size.csv"), "w", encoding="utf-8") as f:
            f.write("filename,pixel size(mm)\n")
            if px is not None:
                f.write(f"{image_name},{px}\n")
            else:
                f.write(f"{image_name},0.15\n")
    else:
        with open(os.path.join(tmp_dir, "pixel_size.csv"), "w", encoding="utf-8") as f:
            f.write("filename,pixel size(mm)\n")
            f.write(f"{image_name},0.15\n")
    return tmp_dir


def run_video_keyframe_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run key-frame + 6-plane classifier in non-interactive agent mode."""
    script = os.path.join(config.agent_tools_dir, "video_keyframe_cls6_predict_agent.py")
    out_dir = _agent_outputs_dir("video_summary", "keyframe_cls6", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--test_script",
            os.path.join(config.keyframe_cls6_dir, "test.py"),
            "--config",
            config.keyframe_cls6_config,
            "--output_csv",
            os.path.join(out_dir, "predictions.csv"),
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        # fetalclip_cls_6_agent/test.py uses Trainer(devices=[3]) internally.
        # main.py may set CUDA_VISIBLE_DEVICES to a single GPU (e.g., "0"),
        # which would hide logical GPU index 3 and crash key-frame inference.
        # Expose full device list for this tool only.
        env_extra={
            "CUDA_VISIBLE_DEVICES": os.environ.get(
                "AGENT_VIDEO_KEYFRAME_VISIBLE_DEVICES",
                "0,1,2,3,4,5,6,7",
            )
        },
        log_prefix="VideoKeyFrame-Cls6",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )
    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_text(result["stdout"])
        for fname, label in kv.items():
            norm = _normalize_video_plane_label(label)
            per_image[fname] = {
                "pred_plane_raw": label,
                "pred_plane_norm": norm,
                "is_key_frame": norm != "other",
            }
    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="VideoKeyFrame-Cls6",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
        artifacts_dir=out_dir,
    )


def _parse_seg_judge_output(text: str) -> Dict[str, str]:
    """
    Parse VLLM judge picks. Accepts lines like:
      - 0041.png — tool1
      0041.png: tool2
      0041.png - none
    Returns: { "0041.png": "tool1" | "tool2" | "none" }
    """
    out: Dict[str, str] = {}
    if not text:
        return out
    rx = re.compile(
        rf"^\s*(?:[-*•]\s*)?(?P<fname>.+\.{_FNAME_EXT_RE})\s*(?:—|–|-|:)\s*(?P<pick>tool\s*1|tool\s*2|tool1|tool2|none)\b",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        m = rx.match(line.strip())
        if not m:
            continue
        fname = m.group("fname").strip()
        pick_raw = m.group("pick").strip().lower().replace(" ", "")
        if pick_raw in ("tool1", "tool2", "none"):
            out[fname] = pick_raw
        elif pick_raw == "tool1":
            out[fname] = "tool1"
        elif pick_raw == "tool2":
            out[fname] = "tool2"
    return out


def _mask_to_raw_array(mask_path: Optional[str], raw_img: Optional[PILImage.Image], preprocess: str) -> Optional[Any]:
    """Rescale predicted binary mask back to the raw image size."""
    if np is None or raw_img is None or not mask_path or not os.path.exists(mask_path):
        return None
    try:
        mask = PILImage.open(mask_path).convert("L")
    except Exception:
        return None

    w, h = raw_img.size
    if preprocess == "resize_direct":
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)
    elif preprocess == "pad_square":
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        mask_sq = mask.resize((max_side, max_side), resample=PILImage.NEAREST)
        mask_on_raw = mask_sq.crop((pad_left, pad_top, pad_left + w, pad_top + h))
    else:
        mask_on_raw = mask.resize((w, h), resample=PILImage.NEAREST)
    arr = np.array(mask_on_raw)  # type: ignore[arg-type]
    return (arr > 0).astype("uint8")


def _load_mask_binary_cv2(mask_path: Optional[str], target_shape: Optional[Tuple[int, int]] = None) -> Optional[Any]:
    """
    Load a binary mask with OpenCV. If target_shape=(h,w), resize with nearest-neighbor.
    """
    if cv2 is None or np is None or not mask_path or not os.path.exists(mask_path):
        return None
    try:
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if target_shape is not None and img.shape[:2] != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (img > 0).astype("uint8")
    except Exception:
        return None


def _dice_masks(a: Optional[Any], b: Optional[Any]) -> Optional[float]:
    if np is None or a is None or b is None:
        return None
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int((a & b).sum())
    return (2.0 * inter) / float(a_sum + b_sum)


def _majority_voting(masks: List[Any]) -> Optional[Any]:
    if np is None:
        return None
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0].copy()
    stacked = np.stack(valid, axis=0)  # type: ignore[arg-type]
    thr = len(valid) / 2.0
    return (stacked.sum(axis=0) >= thr).astype("uint8")


def _keep_largest_component(mask: Any, min_area: int = 50) -> Any:
    if np is None or cv2 is None or mask is None or int(mask.sum()) == 0:
        return mask
    mask_u8 = (mask * 255).astype("uint8")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    if largest_area < min_area:
        return np.zeros_like(mask, dtype="uint8")
    return (labels == largest_idx).astype("uint8")


def _apply_postprocess(mask: Optional[Any], min_area: int = 50) -> Optional[Any]:
    if mask is None:
        return None
    return _keep_largest_component(mask, min_area=min_area)


def _compute_ellipse_residual(mask: Optional[Any]) -> float:
    """Inference-time mask quality score: lower means better ellipse fit."""
    if np is None or cv2 is None or mask is None or int(mask.sum()) == 0:
        return float("nan")
    try:
        mask_bin = (mask > 0).astype("uint8")
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return float("nan")
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return float("nan")
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (major, minor), angle = ellipse
        a = max(major, minor) / 2.0
        b = min(major, minor) / 2.0
        if a <= 0 or b <= 0:
            return float("nan")
        angle_rad = math.radians(angle)
        distances: List[float] = []
        step = max(1, len(contour) // 100)
        for pt in contour[::step]:
            x, y = pt[0]
            dx = float(x) - float(cx)
            dy = float(y) - float(cy)
            x_rot = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
            y_rot = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
            ellipse_val = (x_rot / a) ** 2 + (y_rot / b) ** 2
            dist = abs(ellipse_val - 1.0) * min(a, b)
            distances.append(float(dist))
        return float(sum(distances) / len(distances)) if distances else float("nan")
    except Exception:
        return float("nan")


def _weighted_vote_ensemble_ga(
    ga1: Optional[float],
    ga2: Optional[float],
    ga3: Optional[float],
    tolerance: float = 1.5,
) -> Tuple[Optional[float], str]:
    """
    Weighted vote ensemble (tool2 weight=2.0, tool1/tool3=1.0).
    Returns (final_ga_weeks, source_tag).
    """
    weights = {"tool1": 1.0, "tool2": 2.0, "tool3": 1.0}
    preds: Dict[str, float] = {}
    if ga1 is not None:
        preds["tool1"] = float(ga1)
    if ga2 is not None:
        preds["tool2"] = float(ga2)
    if ga3 is not None:
        preds["tool3"] = float(ga3)
    if not preds:
        return None, "none"
    if len(preds) == 1:
        k, v = next(iter(preds.items()))
        return float(v), f"only_{k}"

    pred_list = list(preds.items())
    best_agreement: Optional[float] = None
    best_weight_sum = 0.0
    best_pair: Optional[str] = None
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            k1, v1 = pred_list[i]
            k2, v2 = pred_list[j]
            if abs(v1 - v2) <= tolerance:
                w1 = weights.get(k1, 1.0)
                w2 = weights.get(k2, 1.0)
                w_sum = w1 + w2
                if w_sum > best_weight_sum:
                    best_weight_sum = w_sum
                    best_pair = f"{k1}_{k2}"
                    best_agreement = (v1 * w1 + v2 * w2) / w_sum

    if best_agreement is not None:
        return float(best_agreement), f"pair_vote_{best_pair}"

    total_w = sum(weights.get(k, 1.0) for k in preds)
    total_v = sum(v * weights.get(k, 1.0) for k, v in preds.items())
    return float(total_v / total_w), "weighted_mean_fallback"


def run_brain_subplane_fetalclip_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Brain subplane classification with FetalCLIP fine-tuned head."""
    script = os.path.join(config.brain_subplane_fetalclip_dir, "predict_agent.py")
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.brain_subplane_fetalclip_ckpt,
        ],
        cwd=config.brain_subplane_fetalclip_dir,
        timeout=config.default_timeout,
        log_prefix="BrainSubplane-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_label_probs(result["stdout"])
        for fname, data in kv.items():
            per_image[fname] = {"subplane": data.get("label"), "probs": data.get("probs")}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="BrainSubplane-FetalCLIP",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_brain_subplane_resnet_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Brain subplane classification with ResNet classifier."""
    script = os.path.join(config.agent_tools_dir, "resnet_cls_predict_agent.py")
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.brain_subplane_resnet_ckpt,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="BrainSubplane-ResNet",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_label_probs(result["stdout"])
        for fname, data in kv.items():
            per_image[fname] = {"subplane": data.get("label"), "probs": data.get("probs")}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="BrainSubplane-ResNet",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_brain_subplane_vit_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Brain subplane classification with ViT classifier."""
    script = os.path.join(config.agent_tools_dir, "vit_cls_predict_agent.py")
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.brain_subplane_vit_ckpt,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="BrainSubplane-ViT",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_label_probs(result["stdout"])
        for fname, data in kv.items():
            per_image[fname] = {"subplane": data.get("label"), "probs": data.get("probs")}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="BrainSubplane-ViT",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_stomach_fetalclip_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "stomach_seg_fetalclip_predict_agent.py")
    out_dir = _agent_outputs_dir("stomach_segmentation", "fetalclip", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.stomach_fetalclip_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="StomachSeg-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="StomachSeg-FetalCLIP",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_stomach_fetalclip_samus_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "stomach_seg_fetalclip_samus_predict_agent.py")
    out_dir = _agent_outputs_dir("stomach_segmentation", "fetalclip_samus", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--fetalclip_ckpt",
            config.stomach_fetalclip_ckpt,
            "--samus_ckpt",
            config.stomach_samus_ckpt,
            "--sam_base",
            config.samus_base_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="StomachSeg-SAMUS",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="StomachSeg-FetalCLIP+SAMUS",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_stomach_nnunet_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    """Run stomach segmentation with nnUNet helper script."""
    script = os.path.join(config.agent_tools_dir, "nnunet_stomach_seg_predict_agent.py")
    out_dir = _agent_outputs_dir("stomach_segmentation", "nnunet", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetal_base_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--out_dir",
            out_dir,
            "--nnunet_predict",
            config.nnunet_predict,
            "--timeout",
            str(config.default_timeout),
            "--progress_every",
            "25",
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout + 120,
        log_prefix="StomachSeg-nnUNet",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            if val and os.path.exists(val):
                per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="StomachSeg-nnUNet",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_abdomen_fetalclip_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "abdomen_seg_fetalclip_predict_agent.py")
    out_dir = _agent_outputs_dir("abdomen_segmentation", "fetalclip", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--ckpt_path",
            config.abdomen_fetalclip_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="AbdomenSeg-FetalCLIP",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="AbdomenSeg-FetalCLIP",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


def run_abdomen_fetalclip_samus_seg_tool(case_dir: str, config: ToolConfig = TOOL_CONFIG) -> ToolResult:
    script = os.path.join(config.agent_tools_dir, "abdomen_seg_fetalclip_samus_predict_agent.py")
    out_dir = _agent_outputs_dir("abdomen_segmentation", "fetalclip_samus", case_dir)
    result = run_tool_subprocess(
        python_path=config.fetalclip2_python,
        script_path=script,
        args=[
            "--data_path",
            case_dir,
            "--fetalclip_ckpt",
            config.abdomen_fetalclip_ckpt,
            "--samus_ckpt",
            config.abdomen_samus_ckpt,
            "--sam_base",
            config.samus_base_ckpt,
            "--out_dir",
            out_dir,
        ],
        cwd=config.agent_tools_dir,
        timeout=config.default_timeout,
        log_prefix="AbdomenSeg-SAMUS",
        print_regexes=[rf"\.{_FNAME_EXT_RE}\s*:\s*"],
    )

    per_image: Dict[str, Dict[str, Any]] = {}
    if result["ok"]:
        kv = _parse_filename_colon_value(result["stdout"])
        for fname, val in kv.items():
            per_image[fname] = {"mask_path": val}

    ok = result["ok"] and len(per_image) > 0
    return ToolResult(
        tool_name="AbdomenSeg-FetalCLIP+SAMUS",
        ok=ok,
        per_image=per_image,
        error=result.get("error") or (None if ok else "No results parsed"),
        logs={"stdout": result["stdout"][-2000:], "stderr": result["stderr"][-2000:]},
    )


# =============================================
# Agent Message Extraction
# =============================================
def extract_agent_text(task_result: Any, agent_name: str) -> str:
    if not task_result or not getattr(task_result, "messages", None):
        return ""
    msgs = [m for m in task_result.messages if getattr(m, "source", None) == agent_name]
    if not msgs:
        for m in reversed(task_result.messages):
            if getattr(m, "type", "").endswith("TextMessage") or getattr(m, "content", None):
                return getattr(m, "content", "")
        return ""
    return getattr(msgs[-1], "content", "")


# =============================================
# Build Agents
# =============================================
def build_agents(model_client: OpenAIChatCompletionClient) -> Dict[str, AssistantAgent]:
    allocator = AssistantAgent(
        name="task_allocator",
        model_client=model_client,
        system_message=(
            "You are the Task Allocation Agent for fetal ultrasound analysis.\n"
            "Input: a user inquiry, image list, and optional preliminary plane-classification results.\n"
            "First decide Inquiry Type as either:\n"
            "  - specific: a targeted request (e.g., estimate GA, measure HC, predict AoP)\n"
            "  - general: broad/comprehensive request (e.g., comprehensive caption, all operations for this plane)\n"
            "Decide which experts to consult from: plane_classification, aop, head_circumference, gestational_age, brain_subplanes, stomach_segmentation, abdomen_segmentation.\n"
            "Rules:\n"
            "  - For specific inquiries, route directly to requested expert(s); do not force plane_classification unless explicitly needed.\n"
            "  - For general inquiries, include plane_classification first, then route plane-dependent experts.\n"
            "  - Respect plane-dependent capabilities summarized in the prompt.\n"
            "  - Output a line: Forwarding to: <comma-separated list>\n"
            "  - Output a line: Inquiry Type: <specific|general>\n"
            "  - After that, include 'Rephrased case:' followed by a concise case description.\n"
            "Example:\n"
            "Forwarding to: gestational_age, head_circumference\n"
            "Inquiry Type: specific\n\n"
            "Rephrased case: Estimate gestational age from the provided fetal brain ultrasound images."
        ),
    )

    aop_expert = AssistantAgent(
        name="aop",
        model_client=model_client,
        system_message=(
            "You are the AoP expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended\": <number|null>, \"recommended_mask_path\": <string|null>, \"decision_note\": <string>}.\n"
            "Do not change recommended values provided by tool-decision logic."
        ),
    )

    hc_expert = AssistantAgent(
        name="head_circumference",
        model_client=model_client,
        system_message=(
            "You are the HC expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended\": <number|null>, \"recommended_mask_path\": <string|null>, \"decision_note\": <string>}.\n"
            "Do not derive GA or add clinical interpretation."
        ),
    )

    ga_expert = AssistantAgent(
        name="gestational_age",
        model_client=model_client,
        system_message=(
            "You are the GA expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended\": {\"weeks\": <int|null>, \"days\": <int|null>}, \"decision_note\": <string>}.\n"
            "Use provided tool-decision values exactly."
        ),
    )

    plane_expert = AssistantAgent(
        name="plane_classification",
        model_client=model_client,
        system_message=(
            "You are the plane classification expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended\": <string>, \"decision_note\": <string>}.\n"
            "Use tool-decision output as source of truth."
        ),
    )

    brain_subplane_expert = AssistantAgent(
        name="brain_subplanes",
        model_client=model_client,
        system_message=(
            "You are the brain subplane expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended\": <string>, \"decision_note\": <string>}.\n"
            "Use tool-decision output as source of truth."
        ),
    )

    stomach_seg_expert = AssistantAgent(
        name="stomach_segmentation",
        model_client=model_client,
        system_message=(
            "You are the stomach segmentation expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended\": <string|null>, \"decision_note\": <string>}.\n"
            "Do not add subjective image interpretation."
        ),
    )

    abdomen_seg_expert = AssistantAgent(
        name="abdomen_segmentation",
        model_client=model_client,
        system_message=(
            "You are the abdomen segmentation expert.\n"
            "Return ONLY valid JSON.\n"
            "Schema per image: {\"recommended_mask_path\": <string|null>, \"recommended_ac_mm\": <number|null>, \"recommended_ga_weeks_from_ac\": <number|null>, \"decision_note\": <string>}.\n"
            "Use tool-decision output as source of truth."
        ),
    )

    summarizer = AssistantAgent(
        name="summarizer",
        model_client=model_client,
        system_message=(
            "You are the Summarizer.\n"
            "Return ONLY valid JSON with this top-level schema:\n"
            "{\n"
            "  \"per_image_reports\": [\n"
            "    {\n"
            "      \"image_name\": \"...\",\n"
            "      \"findings\": {\"standard_plane\": null, \"brain_plane\": null, \"biometry\": {}, \"segmentation\": {}},\n"
            "      \"impression\": {\"estimated_fetal_age\": \"\", \"consistency\": []},\n"
            "      \"comments\": []\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- No subjective clinical interpretation.\n"
            "- Use only structured expert outputs provided.\n"
            "- DO NOT merge different images into one report. One report object per image.\n"
            "- If unavailable, use null or empty string."
        ),
    )

    video_report_generator = AssistantAgent(
        name="video_report_generator",
        model_client=model_client,
        system_message=(
            "You are a dedicated video report generator for continuous fetal ultrasound screenshots.\n"
            "Input is per-frame captions and structured biometry results.\n"
            "Output should be concise and include detected planes, median biometry values, and growth evaluation only.\n"
            "Do not add unsupported clinical interpretation."
        ),
    )

    return {
        "allocator": allocator,
        "aop": aop_expert,
        "head_circumference": hc_expert,
        "gestational_age": ga_expert,
        "plane_classification": plane_expert,
        "brain_subplanes": brain_subplane_expert,
        "stomach_segmentation": stomach_seg_expert,
        "abdomen_segmentation": abdomen_seg_expert,
        "summarizer": summarizer,
        "video_report_generator": video_report_generator,
    }


# =============================================
# Parse Allocator Output
# =============================================
def parse_forwarding_and_rephrased(text: str) -> Tuple[List[str], str, str]:
    forwarding = []
    rephrased = ""
    inquiry_type = "specific"
    t = text.strip()
    
    m = re.search(r"[Ff]orwarding to\s*:\s*([^\n\r]+)", t)
    if m:
        raw = m.group(1)
        parts = re.split(r"[,;]+|\band\b", raw)
        forwarding = [p.strip().lower() for p in parts if p.strip()]
    
    mrep = re.search(r"[Rr]ephrased(?: case)?\s*:\s*(.+)", t, flags=re.DOTALL)
    if mrep:
        rephrased = mrep.group(1).strip()
    else:
        if m:
            rephrased = re.sub(r"[Ff]orwarding to\s*:\s*[^\n\r]+\n?", "", t).strip()
        else:
            rephrased = t

    mt = re.search(r"[Ii]nquiry\s*[Tt]ype\s*:\s*(specific|general)", t)
    if mt:
        inquiry_type = mt.group(1).strip().lower()
    if inquiry_type not in ("specific", "general"):
        inquiry_type = "specific"

    return forwarding, rephrased, inquiry_type


def _enforce_per_image_json(final_text: str) -> str:
    """
    Ensure final output is per-image JSON objects under `per_image_reports`.
    If summarizer merged images into one object, split by image keys when possible.
    """
    try:
        data = json.loads(final_text)
    except Exception:
        return final_text

    if isinstance(data, dict) and isinstance(data.get("per_image_reports"), list):
        return json.dumps(data, ensure_ascii=False, indent=2)

    if not isinstance(data, dict):
        return final_text

    findings = data.get("findings", {})
    if not isinstance(findings, dict):
        return final_text

    # Collect image names from any findings sub-dict keyed by image.
    image_names: set[str] = set()
    for key in ("standard_plane", "brain_plane", "biometry", "segmentation"):
        v = findings.get(key)
        if isinstance(v, dict):
            for k, vv in v.items():
                if isinstance(vv, dict):
                    # nested form: metric -> {image -> value}
                    for kk in vv.keys():
                        if isinstance(kk, str) and kk.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                            image_names.add(kk)
                if isinstance(k, str) and k.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                    image_names.add(k)

    if not image_names:
        return json.dumps({"per_image_reports": []}, ensure_ascii=False, indent=2)

    reports: List[Dict[str, Any]] = []
    for image_name in sorted(image_names):
        std = None
        bp = None
        bio: Dict[str, Any] = {}
        seg: Dict[str, Any] = {}

        sv = findings.get("standard_plane")
        if isinstance(sv, dict) and image_name in sv:
            std = sv.get(image_name)

        bv = findings.get("brain_plane")
        if isinstance(bv, dict) and image_name in bv:
            bp = bv.get(image_name)

        biov = findings.get("biometry")
        if isinstance(biov, dict):
            if image_name in biov and isinstance(biov.get(image_name), dict):
                bio = biov.get(image_name, {})
            else:
                for metric, val in biov.items():
                    if isinstance(val, dict) and image_name in val:
                        bio[metric] = val.get(image_name)

        segv = findings.get("segmentation")
        if isinstance(segv, dict):
            if image_name in segv and isinstance(segv.get(image_name), dict):
                seg = segv.get(image_name, {})
            else:
                for metric, val in segv.items():
                    if isinstance(val, dict) and image_name in val:
                        seg[metric] = val.get(image_name)

        report = {
            "image_name": image_name,
            "patient_information": data.get("patient_information", {"patient_name": "", "date_of_exam": "", "indication": "", "technique": ""}),
            "findings": {
                "standard_plane": std,
                "brain_plane": bp,
                "biometry": bio,
                "segmentation": seg,
            },
            "impression": data.get("impression", {"estimated_fetal_age": "", "consistency": []}),
            "comments": data.get("comments", []),
        }
        reports.append(report)

    return json.dumps({"per_image_reports": reports}, ensure_ascii=False, indent=2)


# =============================================
# Expert Execution Functions
# =============================================
async def run_aop_expert(agent: AssistantAgent, case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run AoP expert with three tools and final outlier-median selection."""
    print(">>> [AoP] Running AoP-SAM tool...")
    result1 = run_aop_sam_tool(case_dir)
    
    print(">>> [AoP] Running USFM-AoP tool...")
    result2 = run_usfm_aop_tool(case_dir)

    print(">>> [AoP] Running UperNet-AoP tool...")
    result3 = run_upernet_aop_tool(case_dir)
    
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()) | set(result3.per_image.keys()))
    final_struct: Dict[str, Any] = {}
    final_predictions: Dict[str, Dict[str, Any]] = {}
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        r3 = result3.per_image.get(fname, {})
        a1 = r1.get("aop_deg")
        a2 = r2.get("aop_deg")
        a3 = r3.get("aop_deg")
        p1 = r1.get("mask_path")
        p2 = r2.get("mask_path")
        p3 = r3.get("mask_path")

        recommended = None
        recommended_mask_path = None
        source = "none"
        note = "No valid tool output"

        if a1 is not None and a2 is not None and a3 is not None:
            vals = [("tool1", float(a1), p1), ("tool2", float(a2), p2), ("tool3", float(a3), p3)]
            sorted_vals = sorted([v for _, v, _ in vals])
            med = float(sorted_vals[1])
            if abs(float(a1) - med) >= 12.0:
                best = min(vals, key=lambda x: abs(x[1] - med))
                source = best[0]
                recommended = best[1]
                recommended_mask_path = best[2]
                note = f"tool1 is outlier (|tool1-median|={abs(float(a1) - med):.2f} >= 12), choose median tool"
            else:
                source = "tool1"
                recommended = float(a1)
                recommended_mask_path = p1
                note = f"tool1 not outlier (|tool1-median|={abs(float(a1) - med):.2f} < 12), keep tool1"
        else:
            # Fallback order when 3-tool rule is not applicable.
            if a1 is not None:
                source = "tool1"
                recommended = float(a1)
                recommended_mask_path = p1
                note = "Fallback to tool1 (3-tool rule unavailable)"
            elif a2 is not None:
                source = "tool2"
                recommended = float(a2)
                recommended_mask_path = p2
                note = "Fallback to tool2 (tool1 unavailable)"
            elif a3 is not None:
                source = "tool3"
                recommended = float(a3)
                recommended_mask_path = p3
                note = "Fallback to tool3 (tool1/tool2 unavailable)"

        final_predictions[fname] = {
            "source": source,
            "recommended_aop_deg": recommended,
            "recommended_mask_path": recommended_mask_path,
            "note": note,
        }
        final_struct[fname] = {
            "recommended": recommended,
            "recommended_mask_path": recommended_mask_path,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "aop",
            "format_version": "1.0",
            "per_image": final_struct,
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "aop",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "final_predictions": final_predictions,
        },
        "expert_text": text,
    }
    

async def run_hc_expert(agent: AssistantAgent, case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run HC expert with CSM + nnUNet and residual-based gating."""
    print(">>> [HC] Running CSM-HC tool...")
    result1 = run_csm_hc_tool(case_dir)
    
    print(">>> [HC] Running HC-nnUNet tool...")
    result2 = run_nnunet_hc_tool(case_dir)

    # Gating aligned with eval_hc_measurement:
    # default nnUNet, switch to CSM only when CSM agrees with nnUNet and has good ellipse residual.
    disagreement_threshold = float(os.environ.get("AGENT_HC_DISAGREEMENT_THRESHOLD", "0.03"))
    residual_threshold = float(os.environ.get("AGENT_HC_RESIDUAL_THRESHOLD", "8.0"))
    final_results: Dict[str, Dict[str, Any]] = {}
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()))
    pixel_map = parse_pixel_size_csv(os.path.join(case_dir, "pixel_size.csv"))
    csm_recomputed_values: Dict[str, Optional[float]] = {}
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        csm_mask_path = r1.get("mask_path")
        nn_mask_path = r2.get("mask_path")

        # Align with eval_hc_measurement.py:
        # - compute residual on native CSM mask
        # - compute HC from masks at original image resolution with Ramanujan-II
        orig_shape: Optional[Tuple[int, int]] = None
        if cv2 is not None:
            case_img = cv2.imread(os.path.join(case_dir, fname), cv2.IMREAD_GRAYSCALE)
            if case_img is not None:
                orig_shape = case_img.shape[:2]

        csm_mask_native = _load_mask_binary_cv2(csm_mask_path, target_shape=None)
        nn_mask_native = _load_mask_binary_cv2(nn_mask_path, target_shape=None)
        csm_mask_hc = _load_mask_binary_cv2(csm_mask_path, target_shape=orig_shape) if orig_shape else csm_mask_native
        nn_mask_hc = _load_mask_binary_cv2(nn_mask_path, target_shape=orig_shape) if orig_shape else nn_mask_native
        pixel_size = pixel_map.get(fname)

        csm_hc = _hc_mm_from_mask_array(csm_mask_hc, pixel_size)
        nn_hc = _hc_mm_from_mask_array(nn_mask_hc, pixel_size)
        csm_recomputed_values[fname] = csm_hc
        # Fallback to tool-emitted values if mask-based recomputation fails.
        if csm_hc is None:
            csm_hc = r1.get("hc_mm")
        if nn_hc is None:
            nn_hc = r2.get("hc_mm")

        csm_residual = _compute_ellipse_residual(csm_mask_native)

        hc_disagreement = float("nan")
        if csm_hc is not None and nn_hc is not None:
            denom = (float(csm_hc) + float(nn_hc)) / 2.0
            if denom != 0:
                hc_disagreement = abs(float(csm_hc) - float(nn_hc)) / denom

        if nn_hc is not None:
            use_csm_guard = (
                csm_hc is not None
                and not math.isnan(hc_disagreement)
                and hc_disagreement < disagreement_threshold
                and not math.isnan(csm_residual)
                and csm_residual < residual_threshold
            )
            if use_csm_guard:
                final_hc = csm_hc
                source = "csm_guard"
            else:
                final_hc = nn_hc
                source = "nnunet_default"
        elif csm_hc is not None:
            final_hc = csm_hc
            source = "csm_fallback"
        else:
            final_hc = None
            source = "none"

        final_results[fname] = {
            "csm_hc_mm": csm_hc,
            "nnunet_hc_mm": nn_hc,
            "csm_ellipse_residual": None if math.isnan(csm_residual) else csm_residual,
            "hc_disagreement": None if math.isnan(hc_disagreement) else hc_disagreement,
            "recommended_hc_mm": final_hc,
            "source": source,
            "csm_mask_path": csm_mask_path,
            "nnunet_mask_path": nn_mask_path,
        }

    # Print corrected CSM values from the aligned Ramanujan pipeline for transparency.
    print(">>> [HC] CSM recomputed (contour + Ramanujan) values:")
    for fname in all_files:
        val = csm_recomputed_values.get(fname)
        if val is None:
            print(f"[HC-CSM-Recomputed] {fname}: N/A")
        else:
            print(f"[HC-CSM-Recomputed] {fname}: {val:.2f} mm")

    structured: Dict[str, Any] = {}
    for fname in all_files:
        fr = final_results.get(fname, {})
        source = fr.get("source")
        csm = fr.get("csm_hc_mm")
        nnv = fr.get("nnunet_hc_mm")
        residual = fr.get("csm_ellipse_residual")
        disagreement = fr.get("hc_disagreement")
        note = "No valid tool output"
        if source == "csm_guard":
            if csm is not None and nnv is not None:
                diff = abs(float(csm) - float(nnv))
                note = f"CSM guard selected (disagreement={disagreement}, residual={residual}); abs diff={diff:.2f} mm"
            else:
                note = "CSM guard selected"
        elif source == "nnunet_default":
            note = "nnUNet default strategy"
        elif source == "csm_fallback":
            note = "nnUNet unavailable; CSM fallback"
        recommended_mask_path = None
        if source in ("nnunet_default",) and fr.get("nnunet_mask_path"):
            recommended_mask_path = fr.get("nnunet_mask_path")
        elif source in ("csm_guard", "csm_fallback") and fr.get("csm_mask_path"):
            recommended_mask_path = fr.get("csm_mask_path")
        structured[fname] = {
            "recommended": _round_1dp(fr.get("recommended_hc_mm")),
            "recommended_mask_path": recommended_mask_path,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "head_circumference",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {
                "type": "nn_with_csm_guard",
                "disagreement_threshold": disagreement_threshold,
                "csm_residual_threshold": residual_threshold,
            },
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "head_circumference",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "final_hc": final_results,
            "gating_thresholds": {
                "disagreement": disagreement_threshold,
                "csm_residual": residual_threshold,
            },
        },
        "expert_text": text,
    }
    

async def run_ga_expert(
    agent: AssistantAgent,
    case_dir: str,
    vignette: str,
    hc_algo_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run GA expert with 3 GA tools + weighted-vote ensemble and HC consistency check."""
    print(">>> [GA] Running GA-RadImageNet tool...")
    result1 = run_ga_algo1_tool(case_dir)
    
    print(">>> [GA] Running GA-FetalCLIP tool...")
    result2 = run_ga_algo2_tool(case_dir)

    print(">>> [GA] Running GA-ConvNeXt tool...")
    result3 = run_ga_algo3_tool(case_dir)

    # Optional HC reference from HC expert output
    hc_per_image_1: Dict[str, Dict[str, Any]] = {}
    hc_per_image_2: Dict[str, Dict[str, Any]] = {}
    if hc_algo_results:
        hc_per_image_1 = (hc_algo_results.get("tool_1") or {}).get("per_image", {}) or {}
        hc_per_image_2 = (hc_algo_results.get("tool_2") or {}).get("per_image", {}) or {}

    all_files = sorted(
        set(result1.per_image.keys())
        | set(result2.per_image.keys())
        | set(result3.per_image.keys())
        | set(hc_per_image_1.keys())
        | set(hc_per_image_2.keys())
    )

    crosscheck: Dict[str, Dict[str, Any]] = {}
    for fname in all_files:
        g1 = result1.per_image.get(fname, {})
        g2 = result2.per_image.get(fname, {})
        g3 = result3.per_image.get(fname, {})
        ga1 = float(g1.get("total_weeks")) if g1.get("total_weeks") is not None else None
        ga2 = float(g2.get("total_weeks")) if g2.get("total_weeks") is not None else None
        ga3 = float(g3.get("total_weeks")) if g3.get("total_weeks") is not None else None

        hc_vals: Dict[str, float] = {}
        if hc_per_image_1.get(fname, {}).get("hc_mm") is not None:
            try:
                hc_vals["hc_tool_1"] = float(hc_per_image_1[fname]["hc_mm"])
            except Exception:
                pass
        if hc_per_image_2.get(fname, {}).get("hc_mm") is not None:
            try:
                hc_vals["hc_tool_2"] = float(hc_per_image_2[fname]["hc_mm"])
            except Exception:
                pass
        hc_ref = (sum(hc_vals.values()) / len(hc_vals)) if hc_vals else None

        def _tool_check(ga_total: Optional[float]) -> Dict[str, Any]:
            if ga_total is None:
                return {"ga_weeks_total": None, "hc_range": None, "in_range": None}
            rng = hc_range_from_ga_weeks(ga_total)
            in_range = None if hc_ref is None else (rng["p2_5"] <= hc_ref <= rng["p97_5"])
            return {"ga_weeks_total": ga_total, "hc_range": rng, "in_range": in_range}

        check1 = _tool_check(ga1)
        check2 = _tool_check(ga2)
        check3 = _tool_check(ga3)

        ens_total, ens_source = _weighted_vote_ensemble_ga(ga1, ga2, ga3, tolerance=1.5)
        ens_weeks = ens_days = None
        ens_range = None
        ens_in_range = None
        if ens_total is not None:
            ens_weeks, ens_days = float_weeks_to_weeks_days(ens_total)
            ens_range = hc_range_from_ga_weeks(ens_total)
            ens_in_range = None if hc_ref is None else (ens_range["p2_5"] <= hc_ref <= ens_range["p97_5"])

        crosscheck[fname] = {
            "hc_values_mm": hc_vals,
            "hc_ref_mm": hc_ref,
            "algo1_check": check1,
            "algo2_check": check2,
            "algo3_check": check3,
            "recommended_ga": {
                "ga_weeks": ens_weeks,
                "ga_days": ens_days,
                "total_weeks": ens_total,
                "source": ens_source,
            },
            "hc_range_from_recommended_ga": ens_range,
            "hc_in_range_for_recommended_ga": ens_in_range,
        }

    structured: Dict[str, Any] = {}
    for fname in all_files:
        cx = crosscheck.get(fname, {})
        rec = cx.get("recommended_ga", {})
        source = str(rec.get("source") or "")
        if source.startswith("pair_vote_"):
            note_base = "Pair-vote agreement between tools"
        elif source == "weighted_mean_fallback":
            note_base = "No close pair; weighted-mean fallback"
        elif source.startswith("only_"):
            note_base = "Single-tool fallback"
        else:
            note_base = "No valid GA decision"
        in_range = cx.get("hc_in_range_for_recommended_ga")
        if in_range is False:
            note = f"{note_base}; HC cross-check out-of-range"
        elif in_range is True:
            note = f"{note_base}; HC cross-check in-range"
        else:
            note = f"{note_base}; HC cross-check unavailable"
        structured[fname] = {
            "recommended": {
                "weeks": rec.get("ga_weeks"),
                "days": rec.get("ga_days"),
            },
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "gestational_age",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {
                "type": "weighted_vote_ensemble",
                "weights": {"tool1": 1.0, "tool2": 2.0, "tool3": 1.0},
                "tolerance_weeks": 1.5,
            },
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "gestational_age",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "hc_crosscheck": crosscheck,
        },
        "expert_text": text,
    }


async def run_plane_expert(agent: AssistantAgent, case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run plane classification expert with two real tools."""
    print(">>> [Plane] Running Plane-FetalCLIP tool...")
    result1 = run_plane_fetalclip_tool(case_dir)
    
    print(">>> [Plane] Running Plane-FU-LoRA tool...")
    result2 = run_plane_fulora_tool(case_dir)
    
    # Normalize plane names
    def normalize_plane(plane: str) -> str:
        plane_lower = (plane or "").lower()
        if "thorax" in plane_lower or "heart" in plane_lower:
            return "thorax"
        if "brain" in plane_lower:
            return "brain"
        if "abdomen" in plane_lower or "kidney" in plane_lower:
            return "abdomen"
        if "femur" in plane_lower:
            return "femur"
        return "other"
    
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()))
    final_results: Dict[str, Dict[str, Any]] = {}
    structured: Dict[str, Any] = {}
    
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        plane1 = normalize_plane(r1.get("plane", "other"))
        plane2 = normalize_plane(r2.get("plane", "other"))
        
        # Decision logic
        if plane1 == plane2:
            final = plane1
            note = "Both algorithms agree"
        elif plane2 == "other" and plane1 != "other":
            final = plane1
            note = "FU-LoRA returned 'other', using FetalCLIP"
        elif plane1 == "other" and plane2 != "other":
            final = plane2
            note = "FetalCLIP returned 'other', using FU-LoRA"
        else:
            final = plane1
            note = "Algorithms disagree, using FetalCLIP"
        
        final_results[fname] = {"plane1": plane1, "plane2": plane2, "final": final, "note": note}
        
        structured[fname] = {
            "recommended": final,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "plane_classification",
            "format_version": "1.0",
            "per_image": structured,
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "plane_classification",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "final_classifications": final_results,
        },
        "expert_text": text,
    }
    

async def run_brain_subplane_expert(agent: AssistantAgent, case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run brain subplane expert with three tools (FetalCLIP, ResNet, ViT)."""
    print(">>> [BrainSubplanes] Running BrainSubplane-FetalCLIP tool...")
    result1 = run_brain_subplane_fetalclip_tool(case_dir)

    print(">>> [BrainSubplanes] Running BrainSubplane-ResNet tool...")
    result2 = run_brain_subplane_resnet_tool(case_dir)

    print(">>> [BrainSubplanes] Running BrainSubplane-ViT tool...")
    result3 = run_brain_subplane_vit_tool(case_dir)

    def norm_label(x: Any) -> str:
        if x is None:
            return ""
        s = str(x).strip()
        if not s:
            return ""
        low = s.lower().replace("_", "-").replace(" ", "")
        if "cerebell" in low:
            return "Trans-cerebellum"
        if "thalam" in low:
            return "Trans-thalamic"
        if "ventric" in low:
            return "Trans-ventricular"
        return s

    final_results: Dict[str, Dict[str, Any]] = {}
    structured: Dict[str, Any] = {}
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()) | set(result3.per_image.keys()))
    for fname in all_files:
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        r3 = result3.per_image.get(fname, {})
        l1 = norm_label(r1.get("subplane"))
        l2 = norm_label(r2.get("subplane"))
        l3 = norm_label(r3.get("subplane"))

        labels = [x for x in [l1, l2, l3] if x]
        final = "N/A"
        note = "No result from any tool"
        if l1 and l2 and l3 and l1 == l2 == l3:
            final = l1
            note = "All three tools agree"
        elif l1 and l2 and l1 == l2:
            final = l1
            note = "Majority: FetalCLIP + ResNet"
        elif l1 and l3 and l1 == l3:
            final = l1
            note = "Majority: FetalCLIP + ViT"
        elif l2 and l3 and l2 == l3:
            final = l2
            note = "Majority: ResNet + ViT"
        elif l1 and l2 and l3 and (l1 != l2 and l1 != l3 and l2 != l3):
            final = l1
            note = "All disagree; default to FetalCLIP"
        elif labels:
            final = labels[0]
            note = "Partial outputs only; using first available prediction"

        final_results[fname] = {
            "tool1": l1 or None,
            "tool2": l2 or None,
            "tool3": l3 or None,
            "final": final,
            "note": note,
        }

        structured[fname] = {
            "recommended": final,
            "decision_note": note,
        }

    text = json.dumps(
        {
            "task": "brain_subplanes",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": "majority_vote_3_tools_else_tool1",
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "brain_subplanes",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "final_classifications": final_results,
        },
        "expert_text": text,
    }
    

async def run_stomach_seg_expert(agent: AssistantAgent, case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run stomach segmentation with 3 tools + tiered fallback shape-prior decision."""
    print(">>> [StomachSeg] Running StomachSeg-FetalCLIP tool...")
    result1 = run_stomach_fetalclip_seg_tool(case_dir)
    print(">>> [StomachSeg] Running StomachSeg-FetalCLIP+SAMUS tool...")
    result2 = run_stomach_fetalclip_samus_seg_tool(case_dir)
    print(">>> [StomachSeg] Running StomachSeg-nnUNet tool...")
    result3 = run_stomach_nnunet_seg_tool(case_dir)

    min_ratio = float(os.environ.get("AGENT_STOMACH_MIN_RATIO", "0.001"))
    min_area = int(os.environ.get("AGENT_STOMACH_MIN_AREA", "50"))

    final_results: Dict[str, Dict[str, Any]] = {}
    all_files = sorted(set(result1.per_image.keys()) | set(result2.per_image.keys()) | set(result3.per_image.keys()))

    for idx, fname in enumerate(all_files):
        r1 = result1.per_image.get(fname, {})
        r2 = result2.per_image.get(fname, {})
        r3 = result3.per_image.get(fname, {})
        p1 = r1.get("mask_path")
        p2 = r2.get("mask_path")
        p3 = r3.get("mask_path")

        raw_path = os.path.join(case_dir, fname)
        raw_img = _safe_load_pil(raw_path)
        m1 = _mask_to_raw_array(p1, raw_img, preprocess="pad_square")
        m2 = _mask_to_raw_array(p2, raw_img, preprocess="resize_direct")
        m3 = _mask_to_raw_array(p3, raw_img, preprocess="resize_direct")
        available = [m for m in [m1, m2, m3] if m is not None]

        final_mask_path = None
        decision_note = "no_mask_available"
        total_pixels = float(raw_img.size[0] * raw_img.size[1]) if raw_img is not None else 1.0
        if raw_img is not None and available:
            base = _majority_voting(available)
            base = _apply_postprocess(base, min_area=min_area)
            base_ratio = 0.0 if base is None else float(base.sum()) / total_pixels
            if base is not None and base_ratio >= min_ratio:
                final_mask_path = p2 or p3 or p1
                decision_note = "majority_minpass"
        if final_mask_path is None:
            r3_ratio = 0.0 if m3 is None else float(m3.sum()) / total_pixels
            r2_ratio = 0.0 if m2 is None else float(m2.sum()) / total_pixels
            r1_ratio = 0.0 if m1 is None else float(m1.sum()) / total_pixels
            if p3 and r3_ratio >= min_ratio:
                final_mask_path = p3
                decision_note = "fallback_tool3"
            elif p2 and r2_ratio >= min_ratio:
                final_mask_path = p2
                decision_note = "fallback_tool2"
            elif p1 and r1_ratio >= min_ratio:
                final_mask_path = p1
                decision_note = "fallback_tool1"
            else:
                final_mask_path = p3 or p2 or p1
                decision_note = "fallback_tool3_default"

        final_results[fname] = {
            "tool1": p1,
            "tool2": p2,
            "tool3": p3,
            "recommended_mask": final_mask_path,
            "note": decision_note,
        }

    structured: Dict[str, Any] = {}
    for fname in all_files:
        fr = final_results.get(fname, {})
        note = str(fr.get("note") or "")
        if note == "majority_minpass":
            cnote = "Majority mask passed shape-prior threshold"
        elif note.startswith("fallback_tool3"):
            cnote = "Fallback to tool3 per tiered rule"
        elif note == "fallback_tool2":
            cnote = "Fallback to tool2 per tiered rule"
        elif note == "fallback_tool1":
            cnote = "Fallback to tool1 per tiered rule"
        else:
            cnote = "Limited/invalid masks; fallback path used"
        structured[fname] = {
            "recommended": fr.get("recommended_mask"),
            "decision_note": cnote,
        }

    text = json.dumps(
        {
            "task": "stomach_segmentation",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {
                "type": "tiered_fallback_shape_prior",
                "min_ratio": min_ratio,
                "min_area": min_area,
                "fallback_order": ["tool3", "tool2", "tool1"],
            },
        },
        ensure_ascii=False,
    )

    return {
        "task": "stomach_segmentation",
        "algo_results": {
            "tool_1": {"name": result1.tool_name, "ok": result1.ok, "per_image": result1.per_image},
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "tool_3": {"name": result3.tool_name, "ok": result3.ok, "per_image": result3.per_image},
            "final_segmentations": final_results,
            "decision_rule": {
                "type": "tiered_fallback_shape_prior",
                "min_ratio": min_ratio,
                "min_area": min_area,
                "fallback_order": ["tool3", "tool2", "tool1"],
            },
        },
        "expert_text": text,
    }


async def run_abdomen_seg_expert(agent: AssistantAgent, case_dir: str, vignette: str) -> Dict[str, Any]:
    """Run abdomen segmentation with tool2 only and compute AC from final masks."""
    print(">>> [AbdomenSeg] Running AbdomenSeg-FetalCLIP+SAMUS tool...")
    result2 = run_abdomen_fetalclip_samus_seg_tool(case_dir)

    # Build image set from both case_dir and tool outputs so missing predictions are visible.
    case_files = [
        n for n in sorted(os.listdir(case_dir))
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))
    ]
    all_files = sorted(set(case_files) | set(result2.per_image.keys()))
    pixel_map = parse_pixel_size_csv(os.path.join(case_dir, "pixel_size.csv"))

    final_results: Dict[str, Dict[str, Any]] = {}
    structured: Dict[str, Any] = {}
    for fname in all_files:
        p2 = (result2.per_image.get(fname) or {}).get("mask_path")
        raw_img = _safe_load_pil(os.path.join(case_dir, fname))
        mask_arr = _mask_to_raw_array(p2, raw_img, preprocess="resize_direct")
        ac_mm = _ac_mm_from_mask_array(mask_arr, pixel_map.get(fname))
        ga_weeks_hadlock = _hadlock_ga_weeks_from_ac_mm(ac_mm)

        final_results[fname] = {
            "tool2_mask_path": p2,
            "ac_mm": ac_mm,
            "ga_weeks_hadlock": ga_weeks_hadlock,
            "pixel_size_mm": pixel_map.get(fname),
            "decision_note": "direct_tool2",
        }

        if p2 and ac_mm is not None and ga_weeks_hadlock is not None:
            cnote = "Directly adopted tool2 mask; AC computed from contour-fitted ellipse; Hadlock GA derived from AC"
        elif p2 and ac_mm is not None:
            cnote = "Directly adopted tool2 mask; AC computed from contour-fitted ellipse; Hadlock GA unavailable"
        elif p2:
            cnote = "Directly adopted tool2 mask; AC unavailable (pixel size or ellipse fit issue)"
        else:
            cnote = "Tool2 mask unavailable"

        structured[fname] = {
            "recommended_mask_path": p2,
            "recommended_ac_mm": _round_1dp(ac_mm),
            "recommended_ga_weeks_from_ac": _round_1dp(ga_weeks_hadlock),
            "decision_note": cnote,
        }

    text = json.dumps(
        {
            "task": "abdomen_segmentation",
            "format_version": "1.0",
            "per_image": structured,
            "decision_rule": {"type": "single_tool_direct", "tool": "AbdomenSeg-FetalCLIP+SAMUS"},
        },
        ensure_ascii=False,
    )
    
    return {
        "task": "abdomen_segmentation",
        "algo_results": {
            "tool_2": {"name": result2.tool_name, "ok": result2.ok, "per_image": result2.per_image},
            "final_segmentations": final_results,
        },
        "expert_text": text,
    }


# =============================================
# Orchestration
# =============================================
async def orchestrate(user_inquiry: str, case_dir: str) -> str:
    """
    Main orchestration function.
    
    Args:
        user_inquiry: User's question/request
        case_dir: Directory containing images and pixel_size.csv
    """
    model_client = build_model_client()
    agents = build_agents(model_client)

    try:
        # Validate case_dir
        if not os.path.isdir(case_dir):
            raise ValueError(f"Case directory does not exist: {case_dir}")
        
        images = [f for f in os.listdir(case_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f">>> Found {len(images)} images in {case_dir}")
        
        if len(images) == 0:
            raise ValueError(f"No images found in {case_dir}")

        if _is_video_summary_request(user_inquiry):
            print(">>> Video-summary request detected. Running key-frame driven workflow...")
            final_text = await run_video_summary_workflow(user_inquiry, case_dir, images, agents)
            print("=" * 60)
            print(">>> FINAL VIDEO REPORT:\n")
            print(final_text)
            print("=" * 60)
            return final_text

        expert_outputs: List[Dict[str, Any]] = []
        plane_summary: Dict[str, Any] = {}

        # Step I: First-pass allocation decides inquiry type (specific/general).
        alloc_prompt_1 = f"""User inquiry: {user_inquiry}

Available images in case directory:
{chr(10).join(['- ' + img for img in images[:20]])}
{"... and more" if len(images) > 20 else ""}

Capabilities:
- Brain plane tasks that are available: brain_subplanes, head_circumference, gestational_age
- Abdomen plane tasks that are available: abdomen_segmentation, stomach_segmentation
- AoP is independent and can be selected directly for specific AoP requests.

Decide inquiry type and initial experts."""

        allocator_res_1 = await agents["allocator"].run(task=alloc_prompt_1)
        allocator_text_1 = extract_agent_text(allocator_res_1, agents["allocator"].name)
        print(f">>> Allocator pass-1 output:\n{allocator_text_1}\n")

        selected, rephrased, inquiry_type = parse_forwarding_and_rephrased(allocator_text_1)
        candidates = {
            "plane_classification",
            "aop",
            "head_circumference",
            "gestational_age",
            "brain_subplanes",
            "stomach_segmentation",
            "abdomen_segmentation",
        }
        selected = [s for s in selected if s in candidates]
        vignette = rephrased if rephrased else user_inquiry

        # General inquiries keep plane-first logic.
        if inquiry_type == "general":
            plane_out = await run_plane_expert(agents["plane_classification"], case_dir, user_inquiry)
            expert_outputs.append(plane_out)
            try:
                plane_json = json.loads(plane_out.get("expert_text", "{}"))
                plane_summary = plane_json.get("per_image", {}) if isinstance(plane_json, dict) else {}
            except Exception:
                plane_summary = {}
            alloc_prompt_2 = f"""User inquiry: {user_inquiry}

Available images in case directory:
{chr(10).join(['- ' + img for img in images[:20]])}
{"... and more" if len(images) > 20 else ""}

Preliminary plane classification result (source of truth):
{json.dumps(plane_summary, ensure_ascii=False)}

Capabilities by plane context:
- Brain plane tasks that are available: brain_subplanes, head_circumference, gestational_age
- Abdomen plane tasks that are available: abdomen_segmentation, stomach_segmentation
- AoP is independent and can be selected when requested by the inquiry.

This is a general inquiry. Decide which experts to consult next and rephrase the case."""
            allocator_res_2 = await agents["allocator"].run(task=alloc_prompt_2)
            allocator_text_2 = extract_agent_text(allocator_res_2, agents["allocator"].name)
            print(f">>> Allocator pass-2 output:\n{allocator_text_2}\n")
            selected2, rephrased2, _ = parse_forwarding_and_rephrased(allocator_text_2)
            selected = [s for s in selected2 if s in candidates and s != "plane_classification"]
            if rephrased2:
                vignette = rephrased2
        else:
            # Specific inquiry: keep direct routing; do not force plane classification.
            selected = [s for s in selected if s != "plane_classification"]

        # GA should consume HC cross-check when GA is requested.
        if "gestational_age" in selected and "head_circumference" not in selected:
            selected.append("head_circumference")

        # Deterministic execution order after plane is done.
        desired_order = [
            "plane_classification",
            "brain_subplanes",
            "aop",
            "stomach_segmentation",
            "abdomen_segmentation",
            "head_circumference",
            "gestational_age",
        ]
        order_index = {name: i for i, name in enumerate(desired_order)}
        seen = set()
        selected = [x for x in selected if not (x in seen or seen.add(x))]
        selected.sort(key=lambda x: order_index.get(x, 999))

        print(f">>> Experts selected (after plane-first allocation): {selected}\n")

        # Step II: Expert execution (remaining experts)
        hc_algo_results_for_ga: Optional[Dict[str, Any]] = None
        
        for name in selected:
            if name == "aop":
                expert_outputs.append(await run_aop_expert(agents["aop"], case_dir, vignette))
            elif name == "plane_classification":
                expert_outputs.append(await run_plane_expert(agents["plane_classification"], case_dir, vignette))
            elif name == "brain_subplanes":
                expert_outputs.append(await run_brain_subplane_expert(agents["brain_subplanes"], case_dir, vignette))
            elif name == "stomach_segmentation":
                expert_outputs.append(await run_stomach_seg_expert(agents["stomach_segmentation"], case_dir, vignette))
            elif name == "abdomen_segmentation":
                expert_outputs.append(await run_abdomen_seg_expert(agents["abdomen_segmentation"], case_dir, vignette))
            elif name == "head_circumference":
                hc_out = await run_hc_expert(agents["head_circumference"], case_dir, vignette)
                hc_algo_results_for_ga = hc_out.get("algo_results")
                expert_outputs.append(hc_out)
            elif name == "gestational_age":
                expert_outputs.append(await run_ga_expert(agents["gestational_age"], case_dir, vignette, hc_algo_results=hc_algo_results_for_ga))

        # Print expert messages
        print(">>> Expert messages to summarizer:")
        for item in expert_outputs:
            print(f"[Expert={item['task']}]\n{item['expert_text']}\n----\n")

        # Step III: Deterministic final report formatting (less JSON-like, template-style).
        final_text = _build_structured_text_summary(
            user_inquiry=user_inquiry,
            images=images,
            expert_outputs=expert_outputs,
        )
        
        print("=" * 60)
        print(">>> FINAL ANSWER:\n")
        print(final_text)
        print("=" * 60)
        
        return final_text
        
    finally:
        try:
            await model_client.close()
        except Exception as e:
            print(f"[ModelClient] close failed: {e}")


async def run_video_summary_workflow(
    user_inquiry: str,
    case_dir: str,
    images: List[str],
    agents: Dict[str, AssistantAgent],
) -> str:
    def _safe_median(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        if np is not None:
            return float(np.median(vals))
        s = sorted(vals)
        n = len(s)
        m = n // 2
        if n % 2 == 1:
            return float(s[m])
        return float((s[m - 1] + s[m]) / 2.0)

    def _frame_caption(case: Dict[str, Any]) -> str:
        fname = case["image_name"]
        plane_raw = case["plane_raw"]
        plane_norm = case["plane_norm"]
        lines: List[str] = []
        if plane_norm == "other":
            lines.append(f"{fname} is classified as No Plane (non-key frame).")
            return " ".join(lines)
        lines.append(f"{fname} shows a clear view of fetal {str(plane_raw).lower()}.")
        if case.get("hc_mm") is not None:
            lines.append(f"Estimated HC is {float(case['hc_mm']):.1f} mm (from {fname}).")
        if case.get("ga_us_weeks") is not None:
            lines.append(f"Estimated GA is {_format_ga_weeks_days(float(case['ga_us_weeks']))} (from {fname}).")
        if case.get("ac_mm") is not None:
            lines.append(f"Estimated AC is {float(case['ac_mm']):.1f} mm (from {fname}).")
        if case.get("ac_ga_weeks") is not None:
            lines.append(
                f"Estimated GA from AC is {_format_ga_weeks_days(float(case['ac_ga_weeks']))} (from {fname})."
            )
        if case.get("growth_note"):
            lines.append(case["growth_note"])
        if plane_norm in ("femur", "thorax", "spine"):
            lines.append("No downstream biometry expert is configured for this plane in current system.")
        return " ".join(lines)

    def _video_report_generator(
        user_inquiry_text: str,
        case_summaries: List[Dict[str, Any]],
        hc_table: List[Dict[str, Any]],
        ac_table: List[Dict[str, Any]],
    ) -> str:
        planes: Dict[str, List[str]] = {}
        hc_vals: List[float] = []
        ga_vals: List[float] = []
        ac_vals: List[float] = []
        ac_ga_vals: List[float] = []
        hc_frames: List[str] = []
        ga_frames: List[str] = []
        ac_frames: List[str] = []
        ac_ga_frames: List[str] = []

        for c in case_summaries:
            if c["plane_norm"] != "other":
                planes.setdefault(c["plane_raw"], []).append(c["image_name"])
            if c.get("hc_mm") is not None:
                hc_vals.append(float(c["hc_mm"]))
                hc_frames.append(c["image_name"])
            if c.get("ga_us_weeks") is not None:
                ga_vals.append(float(c["ga_us_weeks"]))
                ga_frames.append(c["image_name"])
            if c.get("ac_mm") is not None:
                ac_vals.append(float(c["ac_mm"]))
                ac_frames.append(c["image_name"])
            if c.get("ac_ga_weeks") is not None:
                ac_ga_vals.append(float(c["ac_ga_weeks"]))
                ac_ga_frames.append(c["image_name"])

        med_hc = _safe_median(hc_vals)
        med_ga = _safe_median(ga_vals)
        med_ac = _safe_median(ac_vals)
        med_ac_ga = _safe_median(ac_ga_vals)
        lmp_ga_weeks = _extract_lmp_ga_weeks(user_inquiry_text)

        # ---- Build structured report ----
        lines: List[str] = []
        lines.append("Findings:")
        lines.append("")

        # 1) Frame-level plane detection
        lines.append("Frame-level Plane Detection:")
        for c in case_summaries:
            if c["plane_norm"] == "other":
                continue
            lines.append(f"  {c['image_name']} shows a clear view of fetal {str(c['plane_raw']).lower()}.")
        non_key = [c["image_name"] for c in case_summaries if c["plane_norm"] == "other"]
        if non_key:
            lines.append(f"  {len(non_key)} frame(s) classified as non-key (no standard plane detected).")
        lines.append("")

        # Summary of plane counts
        lines.append("Detected Planes Summary:")
        for plane_name, files in sorted(planes.items(), key=lambda kv: kv[0].lower()):
            lines.append(f"  {plane_name}: {len(files)} frame(s)")
        lines.append("")

        # 2) Biometry results
        lines.append("Biometry Results:")
        if med_hc is not None:
            src_txt = ", ".join(hc_frames)
            lines.append(f"  Estimated Head Circumference (HC): {med_hc:.1f} mm"
                         f"{' (median from ' + src_txt + ')' if len(hc_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if med_ga is not None:
            src_txt = ", ".join(ga_frames)
            lines.append(f"  Estimated Gestational Age (GA): {_format_ga_weeks_days(med_ga)}"
                         f"{' (median from ' + src_txt + ')' if len(ga_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if med_ac is not None:
            src_txt = ", ".join(ac_frames)
            lines.append(f"  Estimated Abdomen Circumference (AC): {med_ac:.1f} mm"
                         f"{' (median from ' + src_txt + ')' if len(ac_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if med_ac_ga is not None:
            src_txt = ", ".join(ac_ga_frames)
            lines.append(f"  Estimated GA from AC (Hadlock): {_format_ga_weeks_days(med_ac_ga)}"
                         f"{' (median from ' + src_txt + ')' if len(ac_ga_frames) > 1 else ' (from ' + src_txt + ')'}.")
        if lmp_ga_weeks is not None:
            lines.append(f"  Estimated fetal age {_format_ga_weeks_days(lmp_ga_weeks)} from last menstrual period (LMP).")
        if med_hc is None and med_ga is None and med_ac is None:
            lines.append("  No biometry results available from key frames.")
        lines.append("")

        # 3) Impression & Growth evaluation
        lines.append("Impression:")
        if med_ga is not None:
            lines.append(f"  Estimated fetal age {_format_ga_weeks_days(med_ga)} by ultrasound.")
        elif med_ac_ga is not None:
            lines.append(f"  Estimated fetal age {_format_ga_weeks_days(med_ac_ga)} by ultrasound (from AC/Hadlock).")

        growth_line = None
        if med_hc is not None and med_ga is not None:
            hc_assess = _percentile_assessment(med_hc, med_ga, hc_table)
            if hc_assess:
                if hc_assess["status"] == "within":
                    growth_line = f"HC falls within normal fetal growth range ({hc_assess['band_text']})."
                elif hc_assess["status"] == "larger":
                    growth_line = f"HC is larger than normal fetal growth range (>{hc_assess['normal_text'].split('-')[-1]})."
                else:
                    growth_line = f"HC is smaller than normal fetal growth range (<{hc_assess['normal_text'].split('-')[0]})."
        if growth_line is None and med_ac is not None and lmp_ga_weeks is not None:
            ac_assess = _percentile_assessment(med_ac, lmp_ga_weeks, ac_table)
            if ac_assess:
                if ac_assess["status"] == "within":
                    growth_line = (
                        f"Compared with GA from LMP, AC falls within normal fetal growth range "
                        f"({ac_assess['band_text']})."
                    )
                elif ac_assess["status"] == "larger":
                    growth_line = (
                        f"Compared with GA from LMP, AC is larger than normal fetal growth range "
                        f"(>{ac_assess['normal_text'].split('-')[-1]})."
                    )
                else:
                    growth_line = (
                        f"Compared with GA from LMP, AC is smaller than normal fetal growth range "
                        f"(<{ac_assess['normal_text'].split('-')[0]})."
                    )

        if growth_line:
            lines.append(f"  {growth_line}")
        else:
            lines.append("  Growth evaluation unavailable (insufficient paired biometry/GA evidence).")

        return "\n".join(lines)

    keyframe_res = run_video_keyframe_tool(case_dir)
    if not keyframe_res.ok:
        return "Video summary failed: key-frame detector did not return valid outputs."

    image_to_plane: Dict[str, Dict[str, Any]] = {}
    for pred_name, pdata in keyframe_res.per_image.items():
        matched = _resolve_image_key(pred_name, images)
        if not matched:
            continue
        image_to_plane[matched] = pdata

    for img in images:
        if img not in image_to_plane:
            image_to_plane[img] = {
                "pred_plane_raw": "No Plane",
                "pred_plane_norm": "other",
                "is_key_frame": False,
            }

    hc_ref_path = str(_SCRIPT_DIR / "reference" / "HC_GA_reference.csv")
    ac_ref_path = str(_SCRIPT_DIR / "reference" / "AC_GA_reference.csv")
    hc_table = _load_ga_reference_table(hc_ref_path)
    ac_table = _load_ga_reference_table(ac_ref_path)
    case_summaries: List[Dict[str, Any]] = []
    total_frames = len(images)

    print(f"\n>>> Processing {total_frames} frames individually...")
    print("-" * 60)

    for frame_idx, fname in enumerate(sorted(images), start=1):
        pred = image_to_plane.get(fname, {})
        plane_raw = str(pred.get("pred_plane_raw") or "No Plane")
        plane_norm = str(pred.get("pred_plane_norm") or "other")

        print(f"\n>>> [{frame_idx}/{total_frames}] Processing: {fname}")
        print(f"    Detected plane: {plane_raw}")

        single_dir = _make_single_image_case_dir(case_dir, fname)
        case_item: Dict[str, Any] = {
            "image_name": fname,
            "plane_raw": plane_raw,
            "plane_norm": plane_norm,
            "hc_mm": None,
            "ga_us_weeks": None,
            "ac_mm": None,
            "ac_ga_weeks": None,
            "growth_note": None,
            "caption": "",
        }
        try:
            if plane_norm == "brain":
                print(f"    Experts assigned: head_circumference, gestational_age")
                hc_out = await run_hc_expert(agents["head_circumference"], single_dir, user_inquiry)
                ga_out = await run_ga_expert(
                    agents["gestational_age"],
                    single_dir,
                    user_inquiry,
                    hc_algo_results=hc_out.get("algo_results"),
                )
                parsed = _parse_expert_per_image([hc_out, ga_out])
                hc_map = parsed.get("head_circumference", {})
                ga_map = parsed.get("gestational_age", {})
                hc_entry = next(iter(hc_map.values()), {}) if hc_map else {}
                ga_entry = next(iter(ga_map.values()), {}) if ga_map else {}
                hc_mm = hc_entry.get("recommended")
                ga_rec = ga_entry.get("recommended") or {}
                ga_w = ga_rec.get("weeks")
                ga_d = ga_rec.get("days")
                ga_total = None
                if ga_w is not None and ga_d is not None:
                    ga_total = float(ga_w) + float(ga_d) / 7.0
                if hc_mm is not None:
                    case_item["hc_mm"] = float(hc_mm)
                if ga_total is not None:
                    case_item["ga_us_weeks"] = float(ga_total)

                # Post-hoc HC sanity check: if recommended HC is out of
                # percentile range but the other tool is in range, switch.
                if case_item["hc_mm"] is not None and case_item["ga_us_weeks"] is not None:
                    hc_final_map = (hc_out.get("algo_results") or {}).get("final_hc") or {}
                    hc_detail = next(iter(hc_final_map.values()), {}) if hc_final_map else {}
                    rec_src = hc_detail.get("source", "")
                    csm_val = hc_detail.get("csm_hc_mm")
                    nn_val = hc_detail.get("nnunet_hc_mm")
                    if "csm" in rec_src:
                        alt_val, alt_src = nn_val, "nnunet"
                    else:
                        alt_val, alt_src = csm_val, "csm"
                    checked_hc, checked_src, check_note = _hc_percentile_sanity_check(
                        case_item["hc_mm"], alt_val, case_item["ga_us_weeks"],
                        hc_table, rec_src, alt_src,
                    )
                    if checked_hc is not None and checked_hc != case_item["hc_mm"]:
                        print(f"    [HC sanity check] {check_note}")
                        case_item["hc_mm"] = float(checked_hc)

                if case_item["hc_mm"] is not None and case_item["ga_us_weeks"] is not None:
                    hc_assess = _percentile_assessment(case_item["hc_mm"], case_item["ga_us_weeks"], hc_table)
                    if hc_assess:
                        if hc_assess["status"] == "within":
                            case_item["growth_note"] = f"HC is within normal range ({hc_assess['band_text']})."
                        elif hc_assess["status"] == "larger":
                            case_item["growth_note"] = (
                                f"HC is larger than normal range (>{hc_assess['normal_text'].split('-')[-1]})."
                            )
                        else:
                            case_item["growth_note"] = (
                                f"HC is smaller than normal range (<{hc_assess['normal_text'].split('-')[0]})."
                            )
            elif plane_norm == "abdomen":
                print(f"    Experts assigned: abdomen_segmentation, stomach_segmentation")
                abd_out = await run_abdomen_seg_expert(agents["abdomen_segmentation"], single_dir, user_inquiry)
                sto_out = await run_stomach_seg_expert(agents["stomach_segmentation"], single_dir, user_inquiry)
                parsed = _parse_expert_per_image([abd_out, sto_out])
                abd_map = parsed.get("abdomen_segmentation", {})
                sto_map = parsed.get("stomach_segmentation", {})
                abd_entry = next(iter(abd_map.values()), {}) if abd_map else {}
                sto_entry = next(iter(sto_map.values()), {}) if sto_map else {}
                ac_mm = abd_entry.get("recommended_ac_mm")
                ac_ga = abd_entry.get("recommended_ga_weeks_from_ac")
                if ac_mm is not None:
                    case_item["ac_mm"] = float(ac_mm)
                if ac_ga is not None:
                    case_item["ac_ga_weeks"] = float(ac_ga)
                lmp_ga_weeks = _extract_lmp_ga_weeks(user_inquiry)
                if case_item["ac_mm"] is not None and lmp_ga_weeks is not None:
                    ac_assess = _percentile_assessment(case_item["ac_mm"], lmp_ga_weeks, ac_table)
                    if ac_assess:
                        if ac_assess["status"] == "within":
                            case_item["growth_note"] = (
                                f"Compared with GA from LMP, AC is within normal range ({ac_assess['band_text']})."
                            )
                        elif ac_assess["status"] == "larger":
                            case_item["growth_note"] = (
                                f"Compared with GA from LMP, AC is larger than normal range "
                                f"(>{ac_assess['normal_text'].split('-')[-1]})."
                            )
                        else:
                            case_item["growth_note"] = (
                                f"Compared with GA from LMP, AC is smaller than normal range "
                                f"(<{ac_assess['normal_text'].split('-')[0]})."
                            )
            elif plane_norm in ("femur", "thorax", "spine"):
                print(f"    Experts assigned: none (no biometry/segmentation expert for {plane_raw})")
            else:
                print(f"    Non-key frame; skipping downstream experts.")
        except Exception as e:
            print(f"    [WARNING] Expert execution failed for {fname}: {e}")
        finally:
            shutil.rmtree(single_dir, ignore_errors=True)

        case_item["caption"] = _frame_caption(case_item)
        case_summaries.append(case_item)
        print(f"    Caption: {case_item['caption']}")

    print("\n" + "-" * 60)
    print(f">>> All {total_frames} frames processed. Generating video report...\n")

    report = _video_report_generator(user_inquiry, case_summaries, hc_table, ac_table)
    return report


# =============================================
# CLI
# =============================================
async def _main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Fetal ultrasound multi-agent system")
    parser.add_argument("--inquiry", type=str, required=True, help="User inquiry text")
    parser.add_argument("--case_dir", type=str, required=True, 
                        help="Directory containing images and pixel_size.csv")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Normalize case_dir once at the CLI boundary so subprocess-based tools
    # receive a stable absolute path regardless of their working directory.
    case_dir = os.path.abspath(args.case_dir)

    await orchestrate(args.inquiry, case_dir)


if __name__ == "__main__":
    asyncio.run(_main_cli())
