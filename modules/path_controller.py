# modules/path_controller.py
"""图像路径点到转向控制量的计算模块."""

import numpy as np

import config


class PathController:
    """把图像处理后的路径点转换成单一 steer_signal."""

    def __init__(self):
        self.last_weighted_slope_signal = None
        self.last_weighted_slope_heading_ff = None
        self.last_stanley_lateral_error = None
        self.last_stanley_heading_error = None
        self.last_control_c_lateral_error = None
        self.last_control_c_heading_error = None

    def reset_weighted_slope(self):
        self.last_weighted_slope_signal = None
        self.last_weighted_slope_heading_ff = None

    def reset_stanley_band(self):
        self.last_stanley_lateral_error = None
        self.last_stanley_heading_error = None

    def reset_control_c(self):
        self.last_control_c_lateral_error = None
        self.last_control_c_heading_error = None

    def reset(self):
        self.reset_weighted_slope()
        self.reset_stanley_band()
        self.reset_control_c()

    def compute_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0, lateral_points=None):
        """按配置选择单一转向控制器；各控制器互不自动切换."""
        mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
        if mode == "weighted_slope":
            return self._compute_weighted_steer_signal(
                path_points,
                img_w,
                img_h,
                center_bias_x=center_bias_x,
            )
        if mode == "stanley_band":
            stanley_signal = self._compute_stanley_band_steer_signal(
                path_points,
                img_w,
                img_h,
                center_bias_x=center_bias_x,
                lateral_points=lateral_points,
            )
            if stanley_signal is not None and np.isfinite(stanley_signal):
                return float(stanley_signal)
            return 0.0
        if mode == "control_c":
            control_c_signal = self._compute_control_c_steer_signal(
                path_points,
                img_w,
                img_h,
                center_bias_x=center_bias_x,
                lateral_points=lateral_points,
            )
            if control_c_signal is not None and np.isfinite(control_c_signal):
                return float(control_c_signal)
            return 0.0
        return 0.0

    def select_control_points(self, mode, path_points, img_h):
        """返回当前控制器自己的取样路径点."""
        mode = str(mode).lower()
        if mode == "weighted_slope":
            return self._select_weighted_slope_control_points(path_points, img_h)
        return None

    def control_band_for_mode(self, mode, control_path_points, img_h):
        """返回调试图里表示控制取样位置的 y 范围."""
        if control_path_points is None or len(control_path_points) == 0:
            return None

        mode = str(mode).lower()
        if mode == "control_c":
            sample_rows = [
                float(getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0)),
                float(getattr(
                    config,
                    "CONTROL_C_HEADING_LOOKAHEAD_Y",
                    getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0),
                )),
            ]
            return self._rows_to_band(sample_rows, img_h)

        if mode == "stanley_band":
            sample_rows = [
                float(getattr(config, "STANLEY_LOOKAHEAD_Y", 70.0)),
                float(getattr(
                    config,
                    "STANLEY_HEADING_Y_TOP",
                    getattr(config, "STANLEY_HEADING_FAR_Y", getattr(config, "STANLEY_HEADING_LOOKAHEAD_Y", 70.0)),
                )),
                float(getattr(
                    config,
                    "STANLEY_HEADING_Y_BOTTOM",
                    getattr(config, "STANLEY_HEADING_NEAR_Y", getattr(config, "STANLEY_LOOKAHEAD_Y", 70.0)),
                )),
                float(getattr(
                    config,
                    "STANLEY_FF_Y_TOP",
                    getattr(config, "STANLEY_FF_FAR_Y", getattr(config, "STANLEY_CURVATURE_LOOKAHEAD_Y", 30.0)),
                )),
                float(getattr(
                    config,
                    "STANLEY_FF_Y_BOTTOM",
                    getattr(config, "STANLEY_FF_NEAR_Y", getattr(config, "STANLEY_HEADING_LOOKAHEAD_Y", 70.0)),
                )),
            ]
            return self._rows_to_band(sample_rows, img_h)

        pts = np.array(control_path_points, dtype=np.float32).reshape((-1, 2))
        return (
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 1])),
        )

    def _rows_to_band(self, sample_rows, img_h):
        rows = np.clip(np.array(sample_rows, dtype=np.float32), 0.0, float(img_h) - 1.0)
        return (
            float(np.min(rows)),
            float(np.max(rows)),
        )

    def _compute_weighted_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0):
        """算法 A: 路径点到底部中点连线斜率的加权平均."""
        if path_points is None or len(path_points) == 0:
            self.reset_weighted_slope()
            return 0.0

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        bottom_mid_x = float(img_w) / 2.0 + float(center_bias_x)
        bottom_y = float(img_h) - 1.0
        min_dy = float(config.STEER_SIGNAL_MIN_DY)
        row_gamma = float(getattr(config, "STEER_SIGNAL_ROW_WEIGHT_GAMMA", 1.0))

        dy = np.maximum(bottom_y - pts[:, 1], min_dy)
        slopes = (pts[:, 0] - bottom_mid_x) / dy
        row_weights = np.power(np.clip(pts[:, 1], 0.0, bottom_y), row_gamma)
        weight_sum = float(np.sum(row_weights))
        if weight_sum <= 1e-6:
            self.reset_weighted_slope()
            return 0.0
        slope_signal = float(np.sum(slopes * row_weights) / weight_sum)
        slope_signal *= float(getattr(config, "STEER_SIGNAL_NORMALIZED_SCALE", 1.0))

        ema_alpha = float(getattr(config, "STEER_SIGNAL_D_EMA_ALPHA", 0.65))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 0.98))
        if self.last_weighted_slope_signal is None:
            filtered_signal = float(slope_signal)
            signal_delta = 0.0
        else:
            filtered_signal = (
                ema_alpha * float(self.last_weighted_slope_signal) +
                (1.0 - ema_alpha) * float(slope_signal)
            )
            signal_delta = filtered_signal - float(self.last_weighted_slope_signal)
        self.last_weighted_slope_signal = float(filtered_signal)

        d_gain = float(getattr(config, "STEER_SIGNAL_D_GAIN", 0.0))
        heading_ff = self._compute_weighted_slope_heading_ff(pts, img_h)
        return slope_signal + d_gain * signal_delta + heading_ff

    def _compute_weighted_slope_heading_ff(self, path_points, img_h):
        """算法 A 的小航向前馈：用远/近两行路径方向提前给舵."""
        ff_gain = float(getattr(config, "STEER_SIGNAL_HEADING_FF_GAIN", 0.0))
        if abs(ff_gain) <= 1e-9:
            self.last_weighted_slope_heading_ff = None
            return 0.0

        far_y = float(getattr(config, "STEER_SIGNAL_HEADING_FF_FAR_Y", 35.0))
        near_y = float(getattr(config, "STEER_SIGNAL_HEADING_FF_NEAR_Y", 85.0))
        far_y = float(np.clip(far_y, 0.0, float(img_h) - 1.0))
        near_y = float(np.clip(near_y, 0.0, float(img_h) - 1.0))
        dy = near_y - far_y
        if abs(dy) < 1e-6:
            self.last_weighted_slope_heading_ff = None
            return 0.0

        far_x = self._path_x_at_y_points(path_points, far_y)
        near_x = self._path_x_at_y_points(path_points, near_y)
        if far_x is None or near_x is None:
            self.last_weighted_slope_heading_ff = None
            return 0.0

        heading_slope = (float(far_x) - float(near_x)) / dy
        heading_slope = float(np.clip(heading_slope, -2.0, 2.0))
        scale = float(getattr(config, "STEER_SIGNAL_NORMALIZED_SCALE", 1.0))
        raw_ff = heading_slope * scale * ff_gain

        ema_alpha = float(getattr(config, "STEER_SIGNAL_HEADING_FF_EMA_ALPHA", 0.5))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 0.98))
        if self.last_weighted_slope_heading_ff is None:
            filtered_ff = float(raw_ff)
        else:
            filtered_ff = (
                ema_alpha * float(self.last_weighted_slope_heading_ff) +
                (1.0 - ema_alpha) * float(raw_ff)
            )
        self.last_weighted_slope_heading_ff = float(filtered_ff)
        return float(filtered_ff)

    def _compute_path_control_geometry(
        self,
        path_points,
        img_w,
        img_h,
        *,
        center_bias_x=0.0,
        lateral_points=None,
        lookahead_y=70.0,
        heading_y=None,
        curvature_y=None,
        lateral_half_window=5.0,
        min_fit_points=3,
    ):
        """提取前视横向误差、拟合线航向和曲率，供 C 控制器复用."""
        if path_points is None or len(path_points) == 0:
            return None

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        min_fit_points = max(2, int(min_fit_points))
        if len(pts) < min_fit_points:
            return None

        bottom_mid_x = float(img_w) / 2.0 + float(center_bias_x)
        bottom_y = float(img_h) - 1.0
        lookahead_y = float(np.clip(lookahead_y, 0.0, bottom_y))

        path_x = None
        if lateral_points is not None and len(lateral_points) > 0:
            lateral_pts = np.array(lateral_points, dtype=np.float32).reshape((-1, 2))
            half_window = max(0.0, float(lateral_half_window))
            lateral_mask = np.abs(lateral_pts[:, 1] - lookahead_y) <= half_window
            window_pts = lateral_pts[lateral_mask]
            if len(window_pts) > 0:
                path_x = float(np.mean(window_pts[:, 0]))
        if path_x is None:
            path_x = self._path_x_at_y_points(pts, lookahead_y)
        if path_x is None:
            return None
        lateral_error = float(path_x) - bottom_mid_x

        try:
            poly_coeffs = self._fit_path_poly_coeffs(pts[:, 1], pts[:, 0])
        except Exception:
            return None
        if poly_coeffs is None or len(poly_coeffs) < 3:
            return None

        a = float(poly_coeffs[0])
        b = float(poly_coeffs[1])
        linear_heading_enabled = bool(getattr(config, "PATH_HEADING_LINEAR_FIT_ENABLED", True))
        linear_heading_dx_dy = None
        if linear_heading_enabled:
            try:
                linear_coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
                linear_heading_dx_dy = float(linear_coeffs[0])
            except Exception:
                linear_heading_dx_dy = None
        if heading_y is None:
            heading_y = lookahead_y
        if curvature_y is None:
            curvature_y = heading_y
        heading_y = float(np.clip(float(heading_y), 0.0, bottom_y))
        curvature_y = float(np.clip(float(curvature_y), 0.0, bottom_y))

        heading_dx_dy = linear_heading_dx_dy
        if heading_dx_dy is None:
            heading_dx_dy = 2.0 * a * heading_y + b
        curvature_dx_dy = 2.0 * a * curvature_y + b
        heading_error = float(np.arctan(-heading_dx_dy))
        curvature = float((2.0 * a) / np.power(1.0 + curvature_dx_dy * curvature_dx_dy, 1.5))

        return {
            "lateral_error": float(lateral_error),
            "heading_error": float(heading_error),
            "curvature": float(curvature),
        }

    def _compute_stanley_point_geometry(
        self,
        path_points,
        img_w,
        img_h,
        *,
        center_bias_x=0.0,
        lateral_points=None,
        lookahead_y=70.0,
        heading_y_top=80.0,
        heading_y_bottom=120.0,
        ff_y_top=25.0,
        ff_y_bottom=60.0,
        lateral_half_window=5.0,
        min_fit_points=3,
    ):
        """提取 Stanley 用的横向误差、两点航向角和两点前馈角."""
        if path_points is None or len(path_points) == 0:
            return None

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        min_fit_points = max(2, int(min_fit_points))
        if len(pts) < min_fit_points:
            return None

        bottom_mid_x = float(img_w) / 2.0 + float(center_bias_x)
        bottom_y = float(img_h) - 1.0
        lookahead_y = float(np.clip(lookahead_y, 0.0, bottom_y))

        control_pts = pts

        path_x = None
        half_window = max(0.0, float(lateral_half_window))
        lateral_mask = np.abs(control_pts[:, 1] - lookahead_y) <= half_window
        window_pts = control_pts[lateral_mask]
        if len(window_pts) > 0:
            path_x = float(np.mean(window_pts[:, 0]))
        if path_x is None:
            path_x = self._path_x_at_y_points(control_pts, lookahead_y)
        if path_x is None:
            return None

        heading_error = self._path_angle_between_rows(pts, heading_y_top, heading_y_bottom, bottom_y)
        if heading_error is None:
            return None
        ff_heading_error = self._path_angle_between_rows(pts, ff_y_top, ff_y_bottom, bottom_y)
        if ff_heading_error is None:
            ff_heading_error = heading_error

        return {
            "lateral_error": float(path_x) - bottom_mid_x,
            "heading_error": float(heading_error),
            "ff_heading_error": float(ff_heading_error),
        }

    def _path_angle_between_rows(self, path_points, y_top, y_bottom, bottom_y):
        """用两条 y 行上的路径点估计相对图像竖直方向的角度."""
        y_top = float(np.clip(float(y_top), 0.0, bottom_y))
        y_bottom = float(np.clip(float(y_bottom), 0.0, bottom_y))
        dy = y_bottom - y_top
        if abs(dy) < 1e-6:
            return None

        top_x = self._path_x_at_y_points(path_points, y_top)
        bottom_x = self._path_x_at_y_points(path_points, y_bottom)
        if top_x is None or bottom_x is None:
            return None

        dx_dy = (float(bottom_x) - float(top_x)) / dy
        dx_dy = float(np.clip(dx_dy, -2.0, 2.0))
        return float(np.arctan(-dx_dy))

    def _compute_stanley_band_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0, lateral_points=None):
        """算法 B: 按前视行 Stanley 公式计算转向量."""
        geom = self._compute_stanley_point_geometry(
            path_points,
            img_w,
            img_h,
            center_bias_x=center_bias_x,
            lateral_points=lateral_points,
            lookahead_y=float(getattr(config, "STANLEY_LOOKAHEAD_Y", 70.0)),
            heading_y_top=float(getattr(config, "STANLEY_HEADING_Y_TOP", getattr(config, "STANLEY_HEADING_FAR_Y", getattr(config, "STANLEY_HEADING_LOOKAHEAD_Y", 70.0)))),
            heading_y_bottom=float(getattr(config, "STANLEY_HEADING_Y_BOTTOM", getattr(config, "STANLEY_HEADING_NEAR_Y", getattr(config, "STANLEY_LOOKAHEAD_Y", 70.0)))),
            ff_y_top=float(getattr(config, "STANLEY_FF_Y_TOP", getattr(config, "STANLEY_FF_FAR_Y", getattr(config, "STANLEY_CURVATURE_LOOKAHEAD_Y", 30.0)))),
            ff_y_bottom=float(getattr(config, "STANLEY_FF_Y_BOTTOM", getattr(config, "STANLEY_FF_NEAR_Y", getattr(config, "STANLEY_HEADING_LOOKAHEAD_Y", 70.0)))),
            lateral_half_window=float(getattr(config, "STANLEY_LATERAL_AVG_HALF_WINDOW", 5.0)),
            min_fit_points=int(getattr(config, "STANLEY_MIN_FIT_POINTS", 3)),
        )
        if geom is None:
            self.reset_stanley_band()
            return None

        lateral_error = geom["lateral_error"]
        heading_error = geom["heading_error"]
        ff_heading_error = geom["ff_heading_error"]

        soft = max(1e-6, float(getattr(config, "STANLEY_SOFT", 24.0)))
        speed_estimate = max(0.0, float(getattr(config, "STANLEY_SPEED_ESTIMATE", 0.0)))
        lateral_gain = float(getattr(config, "STANLEY_LATERAL_GAIN", 0.028))
        heading_gain = float(getattr(config, "STANLEY_HEADING_GAIN", 0.85))
        d_gain = float(getattr(config, "STANLEY_LATERAL_D_GAIN", 0.0))
        curvature_gain = float(getattr(config, "STANLEY_CURVATURE_FF_GAIN", 0.0))
        signal_scale = float(getattr(config, "STANLEY_SIGNAL_SCALE", 10000.0))

        ema_alpha = float(getattr(config, "STANLEY_LATERAL_D_EMA_ALPHA", 0.65))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 0.98))
        if self.last_stanley_lateral_error is None:
            filtered_error = float(lateral_error)
            error_delta = 0.0
        else:
            filtered_error = (
                ema_alpha * float(self.last_stanley_lateral_error) +
                (1.0 - ema_alpha) * float(lateral_error)
            )
            error_delta = filtered_error - float(self.last_stanley_lateral_error)
        self.last_stanley_lateral_error = float(filtered_error)

        heading_ema_alpha = float(getattr(config, "STANLEY_HEADING_EMA_ALPHA", 0.5))
        heading_ema_alpha = float(np.clip(heading_ema_alpha, 0.0, 0.98))
        if self.last_stanley_heading_error is None:
            filtered_heading_error = float(heading_error)
        else:
            filtered_heading_error = (
                heading_ema_alpha * float(self.last_stanley_heading_error) +
                (1.0 - heading_ema_alpha) * float(heading_error)
            )
        self.last_stanley_heading_error = float(filtered_heading_error)

        lateral_term = float(np.arctan(lateral_gain * lateral_error / (speed_estimate + soft)))
        d_term = d_gain * error_delta
        heading_term = heading_gain * filtered_heading_error
        curvature_term = curvature_gain * float(ff_heading_error)
        output_sign = float(getattr(config, "STANLEY_OUTPUT_SIGN", 1.0))
        return output_sign * (lateral_term + d_term + heading_term + curvature_term) * signal_scale

    def _compute_control_c_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0, lateral_points=None):
        """算法 C: Kp*e + Kd*de - Kyaw*psi."""
        geom = self._compute_path_control_geometry(
            path_points,
            img_w,
            img_h,
            center_bias_x=center_bias_x,
            lateral_points=lateral_points,
            lookahead_y=float(getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0)),
            heading_y=float(getattr(config, "CONTROL_C_HEADING_LOOKAHEAD_Y", getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0))),
            curvature_y=None,
            lateral_half_window=float(getattr(config, "CONTROL_C_LATERAL_AVG_HALF_WINDOW", 5.0)),
            min_fit_points=int(getattr(config, "CONTROL_C_MIN_FIT_POINTS", 3)),
        )
        if geom is None:
            self.reset_control_c()
            return None

        lateral_error = geom["lateral_error"]
        heading_error = geom["heading_error"]

        lateral_gain = float(getattr(config, "CONTROL_C_LATERAL_GAIN", 0.06))
        heading_gain = float(getattr(config, "CONTROL_C_HEADING_GAIN", 0.5))

        ema_alpha = float(getattr(config, "CONTROL_C_LATERAL_D_EMA_ALPHA", 0.65))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 0.98))
        if self.last_control_c_lateral_error is None:
            filtered_error = float(lateral_error)
            error_delta = 0.0
        else:
            filtered_error = (
                ema_alpha * float(self.last_control_c_lateral_error) +
                (1.0 - ema_alpha) * float(lateral_error)
            )
            error_delta = filtered_error - float(self.last_control_c_lateral_error)
        self.last_control_c_lateral_error = float(filtered_error)

        heading_ema_alpha = float(getattr(config, "CONTROL_C_HEADING_EMA_ALPHA", 0.5))
        heading_ema_alpha = float(np.clip(heading_ema_alpha, 0.0, 0.98))
        if self.last_control_c_heading_error is None:
            filtered_heading_error = float(heading_error)
        else:
            filtered_heading_error = (
                heading_ema_alpha * float(self.last_control_c_heading_error) +
                (1.0 - heading_ema_alpha) * float(heading_error)
            )
        self.last_control_c_heading_error = float(filtered_heading_error)

        lateral_term = lateral_gain * filtered_error
        d_gain = float(getattr(config, "CONTROL_C_LATERAL_D_GAIN", 0.18))
        d_term = d_gain * error_delta
        heading_term = -heading_gain * filtered_heading_error
        return lateral_term + d_term + heading_term

    def _select_weighted_slope_control_points(self, path_points, img_h):
        """算法 A 使用独立行段；若上边界缺失则用最上端点补齐."""
        if path_points is None or len(path_points) == 0:
            return None

        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        row_min = float(getattr(config, "WEIGHTED_SLOPE_SAMPLE_ROW_MIN", 10.0))
        row_max = float(getattr(config, "WEIGHTED_SLOPE_SAMPLE_ROW_MAX", 120.0))
        lo = max(0.0, min(row_min, row_max))
        hi = min(float(img_h) - 1.0, max(row_min, row_max))
        if hi < lo:
            return None

        selected = pts[(pts[:, 1] >= lo) & (pts[:, 1] <= hi)]
        if len(selected) == 0:
            return None
        selected = selected[np.argsort(selected[:, 1])]
        if float(selected[0, 1]) > lo:
            top_pad = np.array([[float(selected[0, 0]), lo]], dtype=np.float32)
            selected = np.vstack([top_pad, selected])
        return selected.astype(np.float32)

    def _path_x_at_y_points(self, path_points, y):
        """在一组路径点上按 y 插值得到 x."""
        pts = np.array(path_points, dtype=np.float32).reshape((-1, 2))
        if len(pts) == 0:
            return None

        order = np.argsort(pts[:, 1])
        ys = pts[order, 1]
        xs = pts[order, 0]
        unique_ys, unique_indices = np.unique(ys, return_index=True)
        unique_xs = xs[unique_indices]
        if len(unique_ys) == 0:
            return None
        if len(unique_ys) == 1:
            return float(unique_xs[0])
        return float(np.interp(float(y), unique_ys, unique_xs))

    def _fit_path_poly_coeffs(self, node_y, node_x):
        """拟合 x=f(y) 的路径多项式，统一二次系数格式."""
        if len(np.unique(node_y)) > 2:
            return np.polyfit(node_y, node_x, 2)

        coeffs = np.polyfit(node_y, node_x, 1)
        return np.insert(coeffs, 0, 0)
