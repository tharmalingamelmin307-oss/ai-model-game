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
        self.last_stanley_debug = None
        self.last_control_c_lateral_error = None
        self.last_control_c_heading_error = None
        self.last_control_c_curve_level = None
        self.last_control_c_debug = None

    def reset_weighted_slope(self):
        self.last_weighted_slope_signal = None
        self.last_weighted_slope_heading_ff = None

    def reset_stanley_band(self):
        self.last_stanley_lateral_error = None
        self.last_stanley_heading_error = None
        self.last_stanley_debug = None

    def reset_control_c(self):
        self.last_control_c_lateral_error = None
        self.last_control_c_heading_error = None
        self.last_control_c_curve_level = None
        self.last_control_c_debug = None

    def reset(self):
        self.reset_weighted_slope()
        self.reset_stanley_band()
        self.reset_control_c()

    def compute_steer_signal(
        self,
        path_points,
        img_w,
        img_h,
        center_bias_x=0.0,
        lateral_points=None,
        heading_points=None,
        d_gain_scale=1.0,
    ):
        """按配置选择单一转向控制器；各控制器互不自动切换."""
        d_gain_scale = float(np.clip(float(d_gain_scale), 0.0, 1.0))
        mode = str(getattr(config, "STEER_CONTROL_MODE", "weighted_slope")).lower()
        if mode == "weighted_slope":
            return self._compute_weighted_steer_signal(
                path_points,
                img_w,
                img_h,
                center_bias_x=center_bias_x,
                d_gain_scale=d_gain_scale,
            )
        if mode == "stanley_band":
            stanley_signal = self._compute_stanley_band_steer_signal(
                path_points,
                img_w,
                img_h,
                center_bias_x=center_bias_x,
                lateral_points=lateral_points,
                heading_points=heading_points,
                d_gain_scale=d_gain_scale,
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
                heading_points=heading_points,
                d_gain_scale=d_gain_scale,
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
                    "CONTROL_C_HEADING_Y_TOP",
                    getattr(config, "CONTROL_C_HEADING_LOOKAHEAD_Y", getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0)),
                )),
                float(getattr(
                    config,
                    "CONTROL_C_HEADING_Y_BOTTOM",
                    getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0),
                )),
                float(getattr(config, "CONTROL_C_FF_Y_TOP", 10.0)),
                float(getattr(config, "CONTROL_C_FF_Y_BOTTOM", 35.0)),
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

    def _compute_weighted_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0, d_gain_scale=1.0):
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

        d_gain = float(getattr(config, "STEER_SIGNAL_D_GAIN", 0.0)) * float(d_gain_scale)
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

    def _compute_stanley_point_geometry(
        self,
        path_points,
        img_w,
        img_h,
        *,
        center_bias_x=0.0,
        lateral_points=None,
        heading_points=None,
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
        lateral_pts = pts
        if lateral_points is not None:
            candidate_lateral_pts = np.array(lateral_points, dtype=np.float32).reshape((-1, 2))
            if len(candidate_lateral_pts) > 0:
                lateral_pts = candidate_lateral_pts

        path_x = None
        half_window = max(0.0, float(lateral_half_window))
        lateral_mask = np.abs(lateral_pts[:, 1] - lookahead_y) <= half_window
        window_pts = lateral_pts[lateral_mask]
        if len(window_pts) > 0:
            distances = np.abs(window_pts[:, 1] - lookahead_y)
            weights = (half_window + 1.0) - distances
            weights = np.maximum(weights, 1e-3)
            weight_sum = float(np.sum(weights))
            if weight_sum > 1e-6:
                path_x = float(np.sum(window_pts[:, 0] * weights) / weight_sum)
        if path_x is None:
            path_x = self._path_x_at_y_points(lateral_pts, lookahead_y)
        if path_x is None:
            return None

        heading_pts = pts
        if heading_points is not None:
            candidate_heading_pts = np.array(heading_points, dtype=np.float32).reshape((-1, 2))
            if len(candidate_heading_pts) >= 2:
                heading_pts = candidate_heading_pts

        heading_error = self._path_angle_between_rows(heading_pts, heading_y_top, heading_y_bottom, bottom_y)
        if heading_error is None:
            return None
        ff_heading_error = self._path_angle_between_rows(heading_pts, ff_y_top, ff_y_bottom, bottom_y)
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
        dx_dy = float(np.clip(dx_dy, -3.5, 3.5))
        return float(np.arctan(-dx_dy))

    def _compute_stanley_band_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0, lateral_points=None, heading_points=None, d_gain_scale=1.0):
        """算法 B: 按前视行 Stanley 公式计算转向量."""
        geom = self._compute_stanley_point_geometry(
            path_points,
            img_w,
            img_h,
            center_bias_x=center_bias_x,
            lateral_points=lateral_points,
            heading_points=heading_points,
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
        d_gain = float(getattr(config, "STANLEY_LATERAL_D_GAIN", 0.0)) * float(d_gain_scale)
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
        signal = output_sign * (lateral_term + d_term + heading_term + curvature_term) * signal_scale
        self.last_stanley_debug = {
            "lateral_error": float(lateral_error),
            "filtered_error": float(filtered_error),
            "error_delta": float(error_delta),
            "heading_error": float(heading_error),
            "filtered_heading_error": float(filtered_heading_error),
            "ff_heading_error": float(ff_heading_error),
            "lateral_gain": float(lateral_gain),
            "d_gain": float(d_gain),
            "heading_gain": float(heading_gain),
            "curvature_gain": float(curvature_gain),
            "soft": float(soft),
            "speed_estimate": float(speed_estimate),
            "signal_scale": float(signal_scale),
            "output_sign": float(output_sign),
            "lateral_term": float(lateral_term),
            "d_term": float(d_term),
            "heading_term": float(heading_term),
            "curvature_term": float(curvature_term),
            "signal": float(signal),
        }
        return signal

    def _compute_control_c_steer_signal(self, path_points, img_w, img_h, center_bias_x=0.0, lateral_points=None, heading_points=None, d_gain_scale=1.0):
        """算法 C: 连续曲率调度控制，e/de 纠偏，psi/psi_ff 顺弯."""
        geom = self._compute_stanley_point_geometry(
            path_points,
            img_w,
            img_h,
            center_bias_x=center_bias_x,
            lateral_points=lateral_points,
            heading_points=heading_points,
            lookahead_y=float(getattr(config, "CONTROL_C_LOOKAHEAD_Y", 70.0)),
            heading_y_top=float(getattr(
                config,
                "CONTROL_C_HEADING_Y_TOP",
                getattr(config, "CONTROL_C_HEADING_LOOKAHEAD_Y", 60.0),
            )),
            heading_y_bottom=float(getattr(
                config,
                "CONTROL_C_HEADING_Y_BOTTOM",
                getattr(config, "CONTROL_C_LOOKAHEAD_Y", 100.0),
            )),
            ff_y_top=float(getattr(config, "CONTROL_C_FF_Y_TOP", 10.0)),
            ff_y_bottom=float(getattr(config, "CONTROL_C_FF_Y_BOTTOM", 35.0)),
            lateral_half_window=float(getattr(config, "CONTROL_C_LATERAL_AVG_HALF_WINDOW", 5.0)),
            min_fit_points=int(getattr(config, "CONTROL_C_MIN_FIT_POINTS", 3)),
        )
        if geom is None:
            self.reset_control_c()
            return None

        lateral_error = geom["lateral_error"]
        heading_error = geom["heading_error"]
        ff_heading_error = geom["ff_heading_error"]

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

        full_heading = max(1e-6, float(getattr(config, "CONTROL_C_CURVE_FULL_HEADING_RAD", 0.35)))
        full_delta = max(1e-6, float(getattr(config, "CONTROL_C_CURVE_FULL_DELTA_RAD", 0.18)))
        curve_from_heading = abs(float(ff_heading_error)) / full_heading
        curve_from_delta = abs(float(ff_heading_error) - float(heading_error)) / full_delta
        raw_curve_level = float(np.clip(max(curve_from_heading, curve_from_delta), 0.0, 1.0))
        curve_alpha = float(getattr(config, "CONTROL_C_CURVE_LEVEL_EMA_ALPHA", 0.85))
        curve_alpha = float(np.clip(curve_alpha, 0.0, 0.98))
        if self.last_control_c_curve_level is None:
            curve_level = raw_curve_level
        else:
            curve_level = (
                curve_alpha * float(self.last_control_c_curve_level) +
                (1.0 - curve_alpha) * raw_curve_level
            )
        self.last_control_c_curve_level = float(curve_level)

        lateral_gain = self._lerp_config(
            "CONTROL_C_LATERAL_GAIN_STRAIGHT",
            "CONTROL_C_LATERAL_GAIN_CURVE",
            float(getattr(config, "CONTROL_C_LATERAL_GAIN", 0.35)),
            float(getattr(config, "CONTROL_C_LATERAL_GAIN", 0.35)),
            curve_level,
        )
        d_gain = self._lerp_config(
            "CONTROL_C_LATERAL_D_GAIN_STRAIGHT",
            "CONTROL_C_LATERAL_D_GAIN_CURVE",
            float(getattr(config, "CONTROL_C_LATERAL_D_GAIN", 0.12)),
            float(getattr(config, "CONTROL_C_LATERAL_D_GAIN", 0.12)),
            curve_level,
        ) * float(d_gain_scale)
        heading_gain = self._lerp_config(
            "CONTROL_C_HEADING_GAIN_STRAIGHT",
            "CONTROL_C_HEADING_GAIN_CURVE",
            float(getattr(config, "CONTROL_C_HEADING_GAIN", 0.0)),
            float(getattr(config, "CONTROL_C_HEADING_GAIN", 0.0)),
            curve_level,
        )
        ff_gain = self._lerp_config(
            "CONTROL_C_FF_GAIN_STRAIGHT",
            "CONTROL_C_FF_GAIN_CURVE",
            float(getattr(config, "CONTROL_C_FF_GAIN", 0.0)),
            float(getattr(config, "CONTROL_C_FF_GAIN", 0.0)),
            curve_level,
        )

        lateral_term = lateral_gain * filtered_error
        d_term = d_gain * error_delta
        heading_term = heading_gain * filtered_heading_error
        ff_term = ff_gain * float(ff_heading_error)
        output_sign = float(getattr(config, "CONTROL_C_OUTPUT_SIGN", 1.0))
        signal = output_sign * (lateral_term + d_term + heading_term + ff_term)
        self.last_control_c_debug = {
            "lateral_error": float(lateral_error),
            "filtered_error": float(filtered_error),
            "error_delta": float(error_delta),
            "heading_error": float(heading_error),
            "filtered_heading_error": float(filtered_heading_error),
            "ff_heading_error": float(ff_heading_error),
            "curve_from_heading": float(curve_from_heading),
            "curve_from_delta": float(curve_from_delta),
            "raw_curve_level": float(raw_curve_level),
            "curve_level": float(curve_level),
            "lateral_gain": float(lateral_gain),
            "d_gain": float(d_gain),
            "heading_gain": float(heading_gain),
            "ff_gain": float(ff_gain),
            "lateral_term": float(lateral_term),
            "d_term": float(d_term),
            "heading_term": float(heading_term),
            "ff_term": float(ff_term),
            "signal": float(signal),
        }
        return signal

    def _lerp_config(self, low_name, high_name, low_default, high_default, t):
        """按 0~1 连续曲率等级在两组参数之间插值."""
        low = float(getattr(config, low_name, low_default))
        high = float(getattr(config, high_name, high_default))
        t = float(np.clip(float(t), 0.0, 1.0))
        return low + (high - low) * t

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
