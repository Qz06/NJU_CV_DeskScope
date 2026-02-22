# pipeline/run_sam3.py
from PIL import Image
import numpy as np
# from sam3.visualization_utils import plot_results  # 不再需要弹窗显示
print("sam3 is ready")


def run_sam3(processor, image, bbox_xyxy, show_result=False):
    """
    运行 SAM3 精细分割
    
    Args:
        processor: SAM3 处理器
        image: PIL Image
        bbox_xyxy: 边界框 [x1, y1, x2, y2]
        show_result: 是否弹窗显示结果（GUI 模式下应设为 False）
    """
    x1, y1, x2, y2 = bbox_xyxy
    W, H = image.size  # PIL Image: (width, height)

    # 像素 → 中心点 + 宽高
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H

    state = processor.set_image(image)
    
    processor.add_geometric_prompt(
        state=state,
        box=[cx, cy, w, h],
        label=True
    )

    def normalize_mask(masks):
        """
        SAM3 masks -> single (H, W) numpy array
        """
        import torch
        import numpy as np

        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()

    # squeeze 所有 1 维
        masks = np.squeeze(masks)

    # 现在只允许 2D 或 3D
        if masks.ndim == 3:
        # 取第一个候选 mask
            masks = masks[0]

        if masks.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")

        return masks

    # SAM3 推理结果已经写回 state
    mask = state["masks"][0]
    mask = normalize_mask(mask)        # torch.Tensor, (H, W)
    
    # 只有在非 GUI 模式下才弹窗显示
    if show_result:
        from sam3.visualization_utils import plot_results
        plot_results(image, state)
    
    return mask
