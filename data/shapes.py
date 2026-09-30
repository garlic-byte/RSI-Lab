"""Draw fresh noisy point clouds with only PyTorch operations."""

import math

import torch


LEGACY_VERSION = "mixed_v1"
DATASET_VERSION = "checkerboard_v2"
LEGACY_BASE_NAMES = ("Checkerboard", "Swiss Roll", "Sine", "Cosine", "Circle", "Two Moons", "Eight Gaussians", "Heart", "Figure Eight", "Star")
BASE_SHAPE_NAMES = tuple(f"Checkerboard {size}x{size}" for size in range(4, 14))
FAMILY_NAMES = ("Ellipse", "Spiral", "Rose", "Polygon", "Star", "Lissajous", "Wave", "Gaussian Ring", "Superellipse")
SHAPE_NAMES = BASE_SHAPE_NAMES + tuple(f"{family} {variant + 1:02d}" for family in FAMILY_NAMES for variant in range(10))


def get_shape_names(dataset_version: str = DATASET_VERSION) -> tuple[str, ...]:
    """Resolve class semantics; unversioned historical checkpoints use mixed_v1."""
    if dataset_version == DATASET_VERSION:
        return SHAPE_NAMES
    if dataset_version == LEGACY_VERSION:
        return LEGACY_BASE_NAMES + SHAPE_NAMES[10:]
    raise ValueError(f"Unsupported dataset_version: {dataset_version}")


def sample_shape(class_id: int, count: int, noise_std: float = 0.035,
                 device: str | torch.device = "cpu", *, dataset_version: str = DATASET_VERSION) -> torch.Tensor:
    """Sample current checkerboard families or explicitly requested legacy data."""
    names = get_shape_names(dataset_version)
    if not 0 <= class_id < len(names):
        raise ValueError(f"class_id must be in 0..{len(names) - 1}")
    if dataset_version == LEGACY_VERSION:
        return sample_legacy_shape(class_id, count, noise_std, device)
    if class_id >= 10:
        points = sample_extended_shape(class_id, count, device)
    else:
        # Same formula as the approved preview: fixed 2.8-wide extent,
        # alternating filled squares, grid size increasing from 4 to 13.
        size = class_id + 4
        cells = torch.tensor([(x, y) for x in range(size) for y in range(size)
                              if (x + y) % 2 == 0], device=device)
        selected = cells[torch.randint(len(cells), (count,), device=device)]
        points = (selected + torch.rand(count, 2, device=device) - size / 2) * (2.8 / size)
    return points + noise_std * torch.randn_like(points)


def sample_extended_shape(class_id: int, count: int, device: str | torch.device) -> torch.Tensor:
    """Nine mathematical families, each with ten distinct parameter settings."""
    family, variant = divmod(class_id - 10, 10)
    u = torch.rand(count, device=device)
    theta = 2 * math.pi * u
    if family == 0:
        ratio = 0.2 + 0.07 * variant
        return torch.stack((1.4 * theta.cos(), 1.4 * ratio * theta.sin()), -1)
    if family == 1:
        angle = (1 + 0.25 * variant) * 2 * math.pi * u
        radius = 0.15 + 1.25 * u
        return torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
    if family == 2:
        radius = 1.35 * torch.cos((variant + 2) * theta)
        return torch.stack((radius * theta.cos(), radius * theta.sin()), -1)
    if family in (3, 4):
        sides = variant + 3
        vertices = sides if family == 3 else 2 * sides
        edge = torch.randint(vertices, (count,), device=device)
        a = edge * (2 * math.pi / vertices) + math.pi / 2
        b = a + 2 * math.pi / vertices
        if family == 3:
            r0 = r1 = torch.full_like(u, 1.35)
        else:
            # Different inner radius from the original five-point star.
            r0 = torch.where(edge % 2 == 0, 1.4, 0.4)
            r1 = torch.where(edge % 2 == 0, 0.4, 1.4)
        start = r0[:, None] * torch.stack((a.cos(), a.sin()), -1)
        end = r1[:, None] * torch.stack((b.cos(), b.sin()), -1)
        return torch.lerp(start, end, u[:, None])
    if family == 5:
        pairs = ((1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (2, 5), (4, 7), (3, 7), (5, 6), (5, 7))
        a, b = pairs[variant]
        return 1.25 * torch.stack((torch.sin(a * theta + math.pi / 4), torch.sin(b * theta)), -1)
    if family == 6:
        x = 3 * u - 1.5
        y = 0.9 * torch.sin((0.5 + 0.25 * variant) * math.pi * x + math.pi / 4)
        return torch.stack((x, y), -1)
    if family == 7:
        centers = variant + 2
        angle = torch.randint(centers, (count,), device=device) * (2 * math.pi / centers)
        points = 1.3 * torch.stack((angle.cos(), angle.sin()), -1)
        return points + 0.045 * torch.randn_like(points)
    # |x/a|^p + |y/b|^p = 1, from concave diamonds to rounded rectangles.
    exponent = 0.6 + 0.45 * variant
    x = theta.cos().sign() * theta.cos().abs().pow(2 / exponent)
    y = theta.sin().sign() * theta.sin().abs().pow(2 / exponent)
    return torch.stack((1.3 * x, 1.05 * y), -1)


def sample_legacy_shape(class_id: int, count: int, noise_std: float = 0.035, device: str | torch.device = "cpu") -> torch.Tensor:
    """Preserve original distributions for evaluating historical checkpoints."""
    if not 0 <= class_id < len(SHAPE_NAMES):
        raise ValueError(f"class_id must be in 0..{len(SHAPE_NAMES) - 1}")
    if class_id >= 10:
        points = sample_extended_shape(class_id, count, device)
        return points + noise_std * torch.randn_like(points)
    u = torch.rand(count, device=device)
    angle = 2 * math.pi * u
    if class_id == 0:
        # Eight alternating filled squares, rather than samples of their edges.
        cells = torch.tensor([(x, y) for x in range(4) for y in range(4) if (x + y) % 2 == 0], device=device)
        points = (cells[torch.randint(8, (count,), device=device)] + torch.rand(count, 2, device=device) - 2) * 0.7
    elif class_id == 1:
        theta = 1.5 * math.pi + 3 * math.pi * u
        radius = theta / (4.5 * math.pi) * 1.4
        points = torch.stack((radius * theta.cos(), radius * theta.sin()), dim=-1)
    elif class_id in (2, 3):
        x = 3 * u - 1.5
        y = torch.sin(math.pi * x) if class_id == 2 else torch.cos(math.pi * x)
        points = torch.stack((x, 0.9 * y), dim=-1)
    elif class_id == 4:
        points = 1.15 * torch.stack((angle.cos(), angle.sin()), dim=-1)
    elif class_id == 5:
        theta = math.pi * u
        side = torch.randint(2, (count,), device=device)
        x = torch.where(side == 0, theta.cos() - 0.5, 0.5 - theta.cos())
        y = torch.where(side == 0, theta.sin() - 0.25, 0.25 - theta.sin())
        points = torch.stack((x, y), dim=-1)
    elif class_id == 6:
        theta = torch.randint(8, (count,), device=device) * (math.pi / 4)
        points = 1.15 * torch.stack((theta.cos(), theta.sin()), dim=-1)
        points = points + 0.075 * torch.randn_like(points)
    elif class_id == 7:
        x = 16 * angle.sin().pow(3)
        y = 13 * angle.cos() - 5 * (2 * angle).cos() - 2 * (3 * angle).cos() - (4 * angle).cos()
        points = torch.stack((x, y), dim=-1) / 12
    elif class_id == 8:
        points = torch.stack((1.3 * angle.sin(), (2 * angle).sin()), dim=-1)
    elif class_id == 9:
        # Uniform samples along each straight segment of a five-point star.
        edge = torch.randint(10, (count,), device=device)
        a = edge * math.pi / 5 + math.pi / 2
        b = (edge + 1) * math.pi / 5 + math.pi / 2
        r0 = torch.where(edge % 2 == 0, 1.4, 0.6)
        r1 = torch.where(edge % 2 == 0, 0.6, 1.4)
        start = r0[:, None] * torch.stack((a.cos(), a.sin()), dim=-1)
        end = r1[:, None] * torch.stack((b.cos(), b.sin()), dim=-1)
        points = torch.lerp(start, end, u[:, None])
    else:
        raise ValueError(f"Unknown class_id: {class_id}; expected 0..9")
    return points + noise_std * torch.randn_like(points)


def sample_batch(count: int, noise_std: float, device: str | torch.device, num_classes: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw class labels uniformly and their corresponding noisy target points."""
    if not 1 <= num_classes <= len(SHAPE_NAMES):
        raise ValueError("num_classes must be in 1..100")
    labels = torch.randint(num_classes, (count,), device=device)
    points = torch.empty(count, 2, device=device)
    for class_id in range(num_classes):
        mask = labels == class_id
        points[mask] = sample_shape(class_id, int(mask.sum()), noise_std, device)
    return points, labels
