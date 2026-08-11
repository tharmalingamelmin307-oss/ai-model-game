// Persistent RKNN segmentation worker for the Python control pipeline.
//
// Protocol (little endian):
//   request:
//     uint32 magic = 'SGIN'
//     uint32 frame_id
//     uint32 payload_bytes
//     uint8[payload_bytes] RGB NHWC model input
//   response:
//     uint32 magic = 'SGOU'
//     uint32 frame_id
//     int32  status
//     uint32 mask_bytes
//     float  run_ms
//     float  post_ms
//     int32  candidates
//     int32  kept
//     uint8[mask_bytes] binary mask, 0/255

#define main bench_rknn_perf_main
#include "bench_rknn_perf.cpp"
#undef main

#include <cerrno>
#include <fstream>
#include <iostream>
#include <string>

namespace {

constexpr std::uint32_t kReqMagic = 0x4E494753U;   // "SGIN"
constexpr std::uint32_t kRespMagic = 0x554F4753U;  // "SGOU"

struct RequestHeader {
  std::uint32_t magic;
  std::uint32_t frame_id;
  std::uint32_t bytes;
};

struct ResponseHeader {
  std::uint32_t magic;
  std::uint32_t frame_id;
  std::int32_t status;
  std::uint32_t mask_bytes;
  float run_ms;
  float post_ms;
  std::int32_t candidates;
  std::int32_t kept;
};

bool read_exact(std::istream &in, void *data, std::size_t bytes) {
  char *ptr = static_cast<char *>(data);
  std::size_t done = 0;
  while (done < bytes) {
    in.read(ptr + done, static_cast<std::streamsize>(bytes - done));
    const std::streamsize got = in.gcount();
    if (got > 0) {
      done += static_cast<std::size_t>(got);
      continue;
    }
    return false;
  }
  return true;
}

bool write_exact(std::ostream &out, const void *data, std::size_t bytes) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(out);
}

std::vector<std::uint8_t> read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open model: " + path);
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("empty model: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  if (!read_exact(file, data.data(), data.size())) {
    throw std::runtime_error("failed to read model: " + path);
  }
  return data;
}

cv::Rect make_output_roi(const candidate_t &candidate) {
  const float x_scale = static_cast<float>(g_mask_output_width) / static_cast<float>(g_input_w);
  const float y_scale = static_cast<float>(g_mask_output_height) / static_cast<float>(g_input_h);
  const float left = std::clamp(candidate.x1, 0.0f, static_cast<float>(g_input_w - 1));
  const float top = std::clamp(candidate.y1, 0.0f, static_cast<float>(g_input_h - 1));
  const float right = std::clamp(candidate.x2, 0.0f, static_cast<float>(g_input_w - 1));
  const float bottom = std::clamp(candidate.y2, 0.0f, static_cast<float>(g_input_h - 1));
  const int x1 = std::clamp(static_cast<int>(left * x_scale), 0, g_mask_output_width);
  const int y1 = std::clamp(static_cast<int>(top * y_scale), 0, g_mask_output_height);
  const int x2 = std::clamp(static_cast<int>(right * x_scale), 0, g_mask_output_width);
  const int y2 = std::clamp(static_cast<int>(bottom * y_scale), 0, g_mask_output_height);
  return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

void or_mask_into_full(const cv::Mat &roi_mask, const cv::Rect &dst, cv::Mat &full_mask) {
  if (roi_mask.empty() || dst.empty()) {
    return;
  }
  const int copy_w = std::min(dst.width, roi_mask.cols);
  const int copy_h = std::min(dst.height, roi_mask.rows);
  for (int y = 0; y < copy_h; ++y) {
    const std::uint8_t *src = roi_mask.ptr<std::uint8_t>(y);
    std::uint8_t *dst_row = full_mask.ptr<std::uint8_t>(dst.y + y) + dst.x;
    for (int x = 0; x < copy_w; ++x) {
      dst_row[x] = static_cast<std::uint8_t>(dst_row[x] | src[x]);
    }
  }
}

bool run_segmentation_tail_to_mask(
    const rknn_output *outs, const rknn_tensor_attr *attrs, const int *mask_candidate_indices,
    int mask_candidate_count, const paddleyolo_rknn::postprocess::Nc1hwc2Int8View *native_proto,
    cv::Mat &full_mask) {
  full_mask = cv::Mat::zeros(g_mask_output_height, g_mask_output_width, CV_8U);
  g_last_mask_pixels = 0;
  g_last_mask_active = 0;
  g_last_proto_roi_area = 0;
  g_last_mask_hash = UINT64_C(1469598103934665603);
  g_last_mask_mode = "none";
  g_last_mask_verify_ms = 0.0;
  if (mask_candidate_count <= 0) {
    return true;
  }

  const int anchors = attrs[2].dims[2];
  const int nm = attrs[2].dims[1];
  const int proto_h = attrs[3].dims[2];
  const int proto_w = attrs[3].dims[3];
  const int proto_pixels = proto_h * proto_w;
  if (native_proto == nullptr || native_proto->data == nullptr) {
    return false;
  }

  float *coeff = ensure(&g_mb, &g_lm, mask_candidate_count * nm);
  const auto *coeff_q = static_cast<const std::int8_t *>(outs[2].buf);
  if (coeff == nullptr || coeff_q == nullptr) {
    g_last_mask_mode = "allocation_failed";
    return false;
  }
  for (int i = 0; i < mask_candidate_count; ++i) {
    const candidate_t &candidate = g_candidates[mask_candidate_indices[i]];
    const int anchor = candidate.anchor;
    for (int channel = 0; channel < nm; ++channel) {
      coeff[static_cast<std::size_t>(i) * nm + channel] = dequant_i8_value(
          coeff_q[static_cast<std::size_t>(channel) * anchors + anchor], attrs[2].zp,
          attrs[2].scale);
    }
    g_last_proto_roi_area += make_proto_roi(candidate, proto_h, proto_w).area();
  }

  const auto decode_path =
      paddleyolo_rknn::postprocess::SelectRoiMaskDecodePath(g_last_proto_roi_area);
  const bool use_int8_roi = decode_path == paddleyolo_rknn::postprocess::RoiMaskDecodePath::kInt8;
  g_last_mask_mode = use_int8_roi ? "roi_tiled_i8" : "roi_tiled_f32";
  float *proto_f32 = nullptr;
  if (!use_int8_roi) {
    proto_f32 = ensure(&g_pb, &g_lp, nm * proto_pixels);
    if (!paddleyolo_rknn::postprocess::DequantizeNc1hwc2Int8ToNchwFloat32(
            *native_proto, attrs[3].scale, attrs[3].zp, proto_f32,
            static_cast<std::size_t>(nm) * proto_pixels)) {
      g_last_mask_mode = "proto_dequant_failed";
      return false;
    }
  }

  for (int i = 0; i < mask_candidate_count; ++i) {
    const candidate_t &candidate = g_candidates[mask_candidate_indices[i]];
    const cv::Rect proto_roi = make_proto_roi(candidate, proto_h, proto_w);
    const cv::Rect output_roi = make_output_roi(candidate);
    if (proto_roi.empty() || output_roi.empty()) {
      continue;
    }

    float *roi_storage = ensure(&g_roi_storage, &g_lroi, proto_roi.area());
    float *resize_storage =
        ensure(&g_resize_storage, &g_lresize, output_roi.width * output_roi.height);
    std::uint8_t *binary_storage =
        ensure_u8(&g_binary_storage, &g_lbinary, output_roi.width * output_roi.height);
    if (roi_storage == nullptr || resize_storage == nullptr || binary_storage == nullptr) {
      g_last_mask_mode = "allocation_failed";
      return false;
    }
    g_roi_logits = cv::Mat(proto_roi.height, proto_roi.width, CV_32F, roi_storage);
    g_mask_resized = cv::Mat(output_roi.height, output_roi.width, CV_32F, resize_storage);
    g_mask_binary = cv::Mat(output_roi.height, output_roi.width, CV_8U, binary_storage);

    const float *coeff_row = coeff + static_cast<std::size_t>(i) * nm;
    if (use_int8_roi) {
      paddleyolo_rknn::postprocess::ComputeRoiMaskInt8Nc1hwc2(
          *native_proto, coeff_row, attrs[3].scale, attrs[3].zp, proto_roi, g_roi_logits,
          paddleyolo_rknn::postprocess::RoiMaskActivation::kSigmoid);
    } else {
      paddleyolo_rknn::postprocess::ComputeRoiMaskFloat32(
          proto_f32, nm, proto_h, proto_w, coeff_row, proto_roi, g_roi_logits,
          paddleyolo_rknn::postprocess::RoiMaskActivation::kSigmoid);
    }
    cv::resize(g_roi_logits, g_mask_resized, output_roi.size(), 0.0, 0.0, cv::INTER_LINEAR);
    paddleyolo_rknn::postprocess::AssignBinaryMaskFromProbabilityMat(g_mask_resized, 127,
                                                                     g_mask_binary);
    or_mask_into_full(g_mask_binary, output_roi, full_mask);
    g_last_mask_pixels += static_cast<long>(g_mask_binary.total());
  }
  return true;
}

double run_staged_seg_postproc_to_mask(
    paddleyolo_rknn::postprocess::FullIoRuntime *runtime, const rknn_tensor_attr *attrs,
    float conf_thr, float iou_thr, int max_det, cv::Mat &full_mask) {
  using paddleyolo_rknn::postprocess::ClassSelectionStats;
  using paddleyolo_rknn::postprocess::ClassifyAnchorsBestClassInt8;
  using paddleyolo_rknn::postprocess::CollectScoreSumSurvivorsInt8;
  using paddleyolo_rknn::postprocess::FullIoTensor;
  using paddleyolo_rknn::postprocess::kScoreSumPrescreenUnavailable;
  using paddleyolo_rknn::postprocess::ScoreSumInt8View;

  const double start = now_ms();
  g_last_candidates = 0;
  g_last_kept = 0;
  g_last_nms_pairs = 0;
  g_last_sum_scanned = 0;
  g_last_score_sum_applied = 0;
  g_last_class_anchors = 0;
  g_last_class_values = 0;
  g_last_fetch_outcome = "pending";
  g_last_staged_failed = false;
  full_mask = cv::Mat::zeros(g_mask_output_height, g_mask_output_width, CV_8U);

  rknn_output *outputs = runtime->Outputs();
  const int anchors = attrs[1].dims[2];
  const int classes = attrs[1].dims[1];
  if (runtime->Prepare(FullIoTensor::kClass) != 0) {
    mark_staged_failure("class_sync_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }

  const int max_candidates = anchors;
  auto *class_seeds = ensure_class_seeds(max_candidates);
  int *survivors = ensure_int(&g_score_sum_survivors, &g_lscore_sum_survivors, anchors);
  if (class_seeds == nullptr || survivors == nullptr) {
    mark_staged_failure("allocation_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }

  int survivor_count = kScoreSumPrescreenUnavailable;
  if (g_score_sum_enabled) {
    if (runtime->Prepare(FullIoTensor::kScoreSum) != 0) {
      mark_staged_failure("score_sum_sync_failed");
      update_full_io_stats(*runtime);
      return now_ms() - start;
    }
    ScoreSumInt8View score_sum{static_cast<const std::int8_t *>(outputs[4].buf),
                               {attrs[4].scale, attrs[4].zp}};
    survivor_count =
        CollectScoreSumSurvivorsInt8(score_sum, anchors, conf_thr, survivors,
                                     static_cast<std::size_t>(anchors));
    if (survivor_count < 0 && survivor_count != kScoreSumPrescreenUnavailable) {
      mark_staged_failure("score_sum_selection_failed");
      update_full_io_stats(*runtime);
      return std::max(0.0, now_ms() - start - g_last_sync_ms);
    }
    if (survivor_count >= 0) {
      g_last_score_sum_applied = 1;
      g_last_sum_scanned = anchors;
      if (survivor_count == 0) {
        g_last_fetch_outcome = "no_score_sum_survivors";
        update_full_io_stats(*runtime);
        return std::max(0.0, now_ms() - start - g_last_sync_ms);
      }
    }
  }

  ClassSelectionStats selection_stats{};
  int seed_count = 0;
  if (survivor_count >= 0) {
    seed_count = ClassifyAnchorsBestClassInt8(
        static_cast<const std::int8_t *>(outputs[1].buf), classes, anchors,
        {attrs[1].scale, attrs[1].zp}, conf_thr, survivors, static_cast<std::size_t>(survivor_count),
        class_seeds, static_cast<std::size_t>(max_candidates), &selection_stats);
    selection_stats.used_score_sum = true;
    selection_stats.score_sum_scanned = anchors;
  } else {
    seed_count = paddleyolo_rknn::postprocess::SelectBestClassSeedsInt8(
        static_cast<const std::int8_t *>(outputs[1].buf), classes, anchors,
        {attrs[1].scale, attrs[1].zp}, conf_thr, nullptr, class_seeds,
        static_cast<std::size_t>(max_candidates), &selection_stats);
  }
  if (seed_count < 0) {
    mark_staged_failure("classification_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  g_last_class_anchors = selection_stats.class_anchors_scanned;
  g_last_class_values = static_cast<long>(selection_stats.class_values_scanned);
  if (seed_count == 0) {
    g_last_fetch_outcome = "no_class_seeds";
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }

  if (runtime->Prepare(FullIoTensor::kBox) != 0) {
    mark_staged_failure("box_sync_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }
  if (decode_candidates_from_seeds(outputs, attrs, true, class_seeds, seed_count, true, iou_thr,
                                   max_det) < 0) {
    mark_staged_failure("candidate_decode_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  if (g_last_kept <= 0) {
    g_last_fetch_outcome = "no_boxes";
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }

  int *mask_candidate_indices =
      ensure_int(&g_mask_candidate_indices, &g_lmask_candidate_indices, g_last_kept);
  const int mask_candidate_count =
      collect_mask_candidate_indices(mask_candidate_indices, g_last_kept);
  if (mask_candidate_count < 0) {
    mark_staged_failure("mask_class_selection_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  if (mask_candidate_count == 0) {
    g_last_fetch_outcome = "no_mask_classes";
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }

  if (runtime->Prepare(FullIoTensor::kMaskCoeff) != 0 ||
      runtime->Prepare(FullIoTensor::kProto) != 0) {
    mark_staged_failure("mask_sync_failed");
    update_full_io_stats(*runtime);
    return now_ms() - start;
  }
  if (!run_segmentation_tail_to_mask(outputs, attrs, mask_candidate_indices, mask_candidate_count,
                                     &runtime->ProtoView(), full_mask)) {
    mark_staged_failure("mask_decode_failed");
    update_full_io_stats(*runtime);
    return std::max(0.0, now_ms() - start - g_last_sync_ms);
  }
  g_last_fetch_outcome = "masks_required";
  update_full_io_stats(*runtime);
  return std::max(0.0, now_ms() - start - g_last_sync_ms);
}

void worker_usage(const char *prog) {
  std::cerr << "usage: " << prog
            << " --model M.rknn [--core 0|1|2|all] [--conf 0.25] [--iou 0.45]\n";
}

}  // namespace

int main(int argc, char **argv) {
  std::string model_path;
  const char *core_str = "0";
  float conf_thr = 0.25f;
  float iou_thr = 0.45f;
  int max_det = 8;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--core" && i + 1 < argc) {
      core_str = argv[++i];
    } else if (arg == "--conf" && i + 1 < argc) {
      conf_thr = std::strtof(argv[++i], nullptr);
    } else if (arg == "--iou" && i + 1 < argc) {
      iou_thr = std::strtof(argv[++i], nullptr);
    } else if (arg == "--max-det" && i + 1 < argc) {
      max_det = std::atoi(argv[++i]);
    } else {
      worker_usage(argv[0]);
      return 2;
    }
  }
  if (model_path.empty()) {
    worker_usage(argv[0]);
    return 2;
  }

  try {
    int core_mask = 0;
    if (!parse_core(core_str, &core_mask)) {
      std::cerr << "invalid core: " << core_str << "\n";
      return 2;
    }
    g_score_sum_enabled = true;
    g_mask_all_classes = false;
    g_mask_class_ids = {0};
    g_mask_output_width = 416;
    g_mask_output_height = 160;

    const auto model = read_file(model_path);
    rknn_context ctx = 0;
    rknn_input_output_num io{};
    rknn_tensor_attr in_attrs[MAX_IO]{};
    rknn_tensor_attr out_attrs[MAX_IO]{};
    void *dummy = nullptr;
    std::size_t input_bytes = 0;
    rknn_input input{};
    paddleyolo_rknn::postprocess::FullIoRuntime full_io;

    const int setup = setup_context(&ctx, model.data(), model.size(), core_mask, 0, 0,
                                    PP_SEG_PREDFL, &io, in_attrs, out_attrs, &dummy,
                                    &input_bytes, &input, &full_io);
    if (setup != 0) {
      std::cerr << "setup_context failed: " << setup << "\n";
      return 1;
    }
    if (g_input_w != 416 || g_input_h != 160) {
      std::cerr << "unexpected seg model input: " << g_input_w << "x" << g_input_h << "\n";
      return 1;
    }

    std::vector<std::uint8_t> frame(input_bytes);
    cv::Mat mask;
    std::cerr << "seg_worker ready input_bytes=" << input_bytes << " mask=416x160\n";

    while (true) {
      RequestHeader req{};
      if (!read_exact(std::cin, &req, sizeof(req))) {
        break;
      }
      ResponseHeader resp{};
      resp.magic = kRespMagic;
      resp.frame_id = req.frame_id;
      resp.status = 0;
      resp.mask_bytes = 416U * 160U;
      if (req.magic != kReqMagic || req.bytes != input_bytes) {
        resp.status = -2;
        resp.mask_bytes = 0;
        write_exact(std::cout, &resp, sizeof(resp));
        std::cout.flush();
        break;
      }
      if (!read_exact(std::cin, frame.data(), frame.size())) {
        break;
      }

      const double t0 = now_ms();
      if (!full_io.SetInput(frame.data(), frame.size())) {
        resp.status = -3;
      } else {
        const int ret = full_io.Run();
        const double t_run = now_ms();
        if (ret != 0) {
          resp.status = ret;
        } else {
          const double post_ms =
              run_staged_seg_postproc_to_mask(&full_io, out_attrs, conf_thr, iou_thr, max_det, mask);
          resp.run_ms = static_cast<float>(t_run - t0);
          resp.post_ms = static_cast<float>(post_ms);
          resp.candidates = g_last_candidates;
          resp.kept = g_last_kept;
          if (staged_postproc_failed()) {
            resp.status = -4;
          }
        }
      }
      if (resp.status != 0 || mask.empty()) {
        resp.mask_bytes = 0;
        write_exact(std::cout, &resp, sizeof(resp));
      } else {
        write_exact(std::cout, &resp, sizeof(resp));
        write_exact(std::cout, mask.data, resp.mask_bytes);
      }
      std::cout.flush();
    }

    full_io.Release();
    if (ctx != 0) {
      rknn_destroy(ctx);
    }
    free(dummy);
    free_postproc_buffers();
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "seg_worker error: " << exc.what() << "\n";
    return 1;
  }
}
