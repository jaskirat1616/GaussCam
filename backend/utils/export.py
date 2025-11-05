"""
Export Utilities

Export Gaussians to various 3D formats (USDZ, OBJ, PLY, GLB, STL, etc.).
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
import pickle

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from pxr import Usd, UsdGeom, Gf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

from backend.gaussian.four_d import Gaussian4D
from backend.compress import compress_gaussians
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


def export_usdz(gaussians, file_path: str) -> bool:
    """
    Export Gaussians to USDZ format (for AR/VR).
    
    Args:
        gaussians: Gaussian object with centroids, colors, scales, rotations
        file_path: Output file path (.usdz)
    
    Returns:
        True if successful
    """
    if not USD_AVAILABLE:
        raise ImportError(
            "USD library required for USDZ export. "
            "Install with: pip install usd-core"
        )
    
    try:
        # Create USD stage
        stage = Usd.Stage.CreateNew(file_path)
        
        # Get Gaussian data (use centroids as positions)
        positions = gaussians.centroids  # (N, 3)
        colors = gaussians.colors if hasattr(gaussians, 'colors') else None  # (N, 3)
        scales = gaussians.scales if hasattr(gaussians, 'scales') else None  # (N, 3) or (N,)
        opacities = gaussians.opacity if hasattr(gaussians, 'opacity') else np.ones(len(positions))
        
        # Create point cloud mesh
        points_prim = UsdGeom.Points.Define(stage, "/Gaussians")
        
        # Set positions
        points_prim.CreatePointsAttr().Set([Gf.Vec3f(p[0], p[1], p[2]) for p in positions])
        
        # Set colors
        if colors is not None and len(colors) > 0:
            color_primvar = points_prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            # Convert colors from [0, 1] to [0, 255] if needed
            if colors.max() <= 1.0:
                colors_normalized = colors
            else:
                colors_normalized = colors / 255.0
            color_primvar.Set([Gf.Vec3f(c[0], c[1], c[2]) for c in colors_normalized])
        
        # Set widths (using scales)
        if scales is not None and len(scales) > 0:
            widths = np.mean(scales, axis=1) if scales.ndim > 1 else scales
            points_prim.CreateWidthsAttr().Set(widths.tolist())
        
        # Save as USDZ
        stage.Export(file_path)
        logger.info(f"Exported {len(positions)} Gaussians to USDZ: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export USDZ: {e}", exc_info=True)
        raise


def export_ply(gaussians, file_path: str) -> bool:
    """
    Export Gaussians to PLY format.
    
    Args:
        gaussians: Gaussian object
        file_path: Output file path (.ply)
    
    Returns:
        True if successful
    """
    try:
        positions = gaussians.centroids
        colors = gaussians.colors if hasattr(gaussians, 'colors') else None
        
        # Write PLY header
        with open(file_path, 'wb') as f:
            # Header
            num_vertices = len(positions)
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_vertices}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write vertices
            for i, pos in enumerate(positions):
                x, y, z = pos
                if colors is not None and i < len(colors):
                    r, g, b = colors[i]
                    # Convert to [0, 255]
                    if r <= 1.0:
                        r, g, b = int(r * 255), int(g * 255), int(b * 255)
                    else:
                        r, g, b = int(r), int(g), int(b)
                else:
                    r, g, b = 128, 128, 128
                
                # Write binary data
                f.write(np.array([x, y, z], dtype=np.float32).tobytes())
                f.write(np.array([r, g, b], dtype=np.uint8).tobytes())
        
        logger.info(f"Exported {len(positions)} Gaussians to PLY: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export PLY: {e}", exc_info=True)
        raise


def export_obj(gaussians, file_path: str) -> bool:
    """
    Export Gaussians to OBJ format (as point cloud).
    
    Args:
        gaussians: Gaussian object
        file_path: Output file path (.obj)
    
    Returns:
        True if successful
    """
    try:
        positions = gaussians.centroids
        colors = gaussians.colors if hasattr(gaussians, 'colors') else None
        
        with open(file_path, 'w') as f:
            # Write header
            f.write("# Gaussian Splatting Point Cloud\n")
            f.write(f"# {len(positions)} points\n\n")
            
            # Write vertices
            for i, pos in enumerate(positions):
                x, y, z = pos
                if colors is not None and i < len(colors):
                    r, g, b = colors[i]
                    # OBJ doesn't support vertex colors directly, but we can use comments
                    # For now, just write positions
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                else:
                    f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        
        logger.info(f"Exported {len(positions)} Gaussians to OBJ: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export OBJ: {e}", exc_info=True)
        raise


def export_glb(gaussians, file_path: str) -> bool:
    """
    Export Gaussians to GLB format (glTF binary).
    
    Args:
        gaussians: Gaussian object
        file_path: Output file path (.glb)
    
    Returns:
        True if successful
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError(
            "trimesh required for GLB export. Install with: pip install trimesh"
        )
    
    try:
        positions = gaussians.centroids
        colors = gaussians.colors if hasattr(gaussians, 'colors') else None
        
        # Create point cloud
        if colors is not None and len(colors) > 0:
            # Convert colors to [0, 255]
            if colors.max() <= 1.0:
                colors_uint8 = (colors * 255).astype(np.uint8)
            else:
                colors_uint8 = colors.astype(np.uint8)
            
            point_cloud = trimesh.PointCloud(
                vertices=positions,
                colors=colors_uint8
            )
        else:
            point_cloud = trimesh.PointCloud(vertices=positions)
        
        # Export to GLB
        point_cloud.export(file_path)
        logger.info(f"Exported {len(positions)} Gaussians to GLB: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export GLB: {e}", exc_info=True)
        raise


def export_gltf_animation(gaussians: Gaussian4D, file_path: str) -> bool:
    """Export dynamic Gaussians to a multi-frame glTF scene."""

    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for glTF export. Install with: pip install trimesh")

    try:
        base_vertices = gaussians.gaussian.centroids
        motion = gaussians.motion.translation
        colors = gaussians.gaussian.colors

        scene = trimesh.Scene()
        frame_vertices = [base_vertices, base_vertices + motion]

        for idx, verts in enumerate(frame_vertices):
            color_uint8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
            mesh = trimesh.PointCloud(vertices=verts, colors=color_uint8)
            scene.add_geometry(mesh, node_name=f"frame_{idx}")

        scene.export(file_path)
        logger.info(f"Exported dynamic Gaussians to glTF: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export glTF animation: {e}", exc_info=True)
        raise


def export_compressed(gaussians, file_path: str, target_count: Optional[int] = None) -> bool:
    """Export compressed Gaussians using quantization and codebooks."""

    importance = np.ones(gaussians.num_gaussians, dtype=np.float32)
    compressed = compress_gaussians(
        gaussians,
        importance_scores=importance,
        target_count=target_count,
    )

    quantized = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in compressed["quantized"].items()
    }

    payload = {
        "quantized": quantized,
        "codebook": {
            "codebook": compressed["codebook"]["codebook"].tolist(),
            "assignments": compressed["codebook"]["assignments"].tolist(),
        },
        "stats": compressed["stats"].__dict__,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    logger.info(f"Exported compressed Gaussians to {file_path}")
    return True


def export_stl(gaussians, file_path: str) -> bool:
    """
    Export Gaussians to STL format (as point cloud converted to mesh).
    
    Args:
        gaussians: Gaussian object
        file_path: Output file path (.stl)
    
    Returns:
        True if successful
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError(
            "trimesh required for STL export. Install with: pip install trimesh"
        )
    
    try:
        positions = gaussians.centroids
        
        # Create point cloud and convert to mesh using ball pivoting
        point_cloud = trimesh.PointCloud(vertices=positions)
        
        # Try to create mesh from point cloud
        try:
            mesh = point_cloud.convex_hull
            mesh.export(file_path)
        except:
            # If convex hull fails, export as point cloud (will be empty STL)
            # Better to export as PLY instead
            logger.warning("STL export requires mesh conversion. Consider using PLY format instead.")
            raise ValueError("STL format requires mesh conversion. Use PLY or OBJ for point clouds.")
        
        logger.info(f"Exported {len(positions)} Gaussians to STL: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export STL: {e}", exc_info=True)
        raise


def export_gaussians(gaussians, file_path: str, format: str = "auto") -> bool:
    """
    Export Gaussians to specified format.
    
    Args:
        gaussians: Gaussian object
        file_path: Output file path
        format: Export format ('usdz', 'ply', 'obj', 'glb', 'stl', 'pkl', 'auto')
    
    Returns:
        True if successful
    """
    file_path = Path(file_path)
    
    # Auto-detect format from extension
    if format == "auto":
        format = file_path.suffix.lower().lstrip('.')
    
    format = format.lower()
    
    if format == "usdz":
        return export_usdz(gaussians, str(file_path))
    elif format == "ply":
        return export_ply(gaussians, str(file_path))
    elif format == "obj":
        return export_obj(gaussians, str(file_path))
    elif format == "glb":
        return export_glb(gaussians, str(file_path))
    elif format == "gltf":
        if not isinstance(gaussians, Gaussian4D):
            raise TypeError("glTF export requires Gaussian4D input")
        return export_gltf_animation(gaussians, str(file_path))
    elif format == "gca":
        return export_compressed(gaussians, str(file_path))
    elif format == "stl":
        return export_stl(gaussians, str(file_path))
    elif format in ["pkl", "pickle"]:
        # Pickle export
        with open(file_path, 'wb') as f:
            pickle.dump(gaussians, f)
        logger.info(f"Exported Gaussians to pickle: {file_path}")
        return True
    else:
        raise ValueError(f"Unsupported export format: {format}")

