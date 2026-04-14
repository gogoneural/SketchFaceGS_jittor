import jittor as jt


def axis_angle_to_quaternion(axis_angle: jt.Var) -> jt.Var:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = jt.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = (angles.abs() < eps)
    # for x small, sin(x/2)/x is about 1/2 - (x*x)/48
    small_val = 0.5 - (angles * angles) / 48
    # for normal angles: sin(x/2) / x
    safe_angles = jt.clamp(angles, min_v=eps)  # avoid division by zero
    normal_val = jt.sin(half_angles) / safe_angles
    sin_half_angles_over_angles = jt.ternary(small_angles, small_val, normal_val)
    quaternions = jt.concat(
        [jt.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions