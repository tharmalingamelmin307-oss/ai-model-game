// Persistent RKNN detection worker for YOLOv8 predfl models.

#define main bench_rknn_perf_main
#include "bench_rknn_perf.cpp"
#undef main

#include <fstream>
#include <iostream>
#include <string>

namespace {

constexpr std::uint32_t kReqMagic = 0x4E494459U;   // "YDIN"
constexpr std::uint32_t kRespMagic = 0x554F4459U;  // "YDOU"

struct RequestHeader {
  std::uint32_t magic;
  std::uint32_t frame_id;
  std::uint32_t bytes;
};

struct ResponseHeader {
  std::uint32_t magic;
  std::uint32_t frame_id;
  std::int32_t status;
  std::uint32_t count;
  float run_ms;
  float post_ms;
  std::int32_t candidates;
  std::int32_t kept;
};

struct DetectionRecord {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  std::int32_t cls;
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

void worker_usage(const char *prog) {
  std::cerr << "usage: " << prog
            << " --model M.rknn [--core 0|1|2|all] [--conf 0.5] [--iou 0.45]"
               " [--max-det 50]\n";
}

}  // namespace

int main(int argc, char **argv) {
  std::string model_path;
  const char *core_str = "2";
  float conf_thr = 0.5f;
  float iou_thr = 0.45f;
  int max_det = 50;

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

    const auto model = read_file(model_path);
    rknn_context ctx = 0;
    rknn_input_output_num io{};
    rknn_tensor_attr in_attrs[MAX_IO]{};
    rknn_tensor_attr out_attrs[MAX_IO]{};
    void *dummy = nullptr;
    std::size_t input_bytes = 0;
    rknn_input input{};
    paddleyolo_rknn::postprocess::FullIoRuntime full_io;

    const int setup = setup_context(&ctx, model.data(), model.size(), core_mask, 0, 0, PP_PREDFL,
                                    &io, in_attrs, out_attrs, &dummy, &input_bytes, &input,
                                    &full_io);
    if (setup != 0) {
      std::cerr << "setup_context failed: " << setup << "\n";
      return 1;
    }
    if (g_input_w != 640 || g_input_h != 640) {
      std::cerr << "unexpected det model input: " << g_input_w << "x" << g_input_h << "\n";
      return 1;
    }

    std::vector<std::uint8_t> frame(input_bytes);
    std::vector<DetectionRecord> records;
    records.reserve(static_cast<std::size_t>(max_det));
    std::cerr << "det_worker ready input_bytes=" << input_bytes << "\n";

    while (true) {
      RequestHeader req{};
      if (!read_exact(std::cin, &req, sizeof(req))) {
        break;
      }
      ResponseHeader resp{};
      resp.magic = kRespMagic;
      resp.frame_id = req.frame_id;
      resp.status = 0;
      if (req.magic != kReqMagic || req.bytes != input_bytes) {
        resp.status = -2;
        write_exact(std::cout, &resp, sizeof(resp));
        std::cout.flush();
        break;
      }
      if (!read_exact(std::cin, frame.data(), frame.size())) {
        break;
      }

      input.buf = frame.data();
      input.size = input_bytes;
      const double t0 = now_ms();
      int ret = rknn_inputs_set(ctx, 1, &input);
      if (ret == 0) {
        ret = rknn_run(ctx, nullptr);
      }
      const double t_run = now_ms();
      rknn_output outs[MAX_IO];
      std::memset(outs, 0, sizeof(outs));
      if (ret == 0) {
        for (std::uint32_t o = 0; o < io.n_output; ++o) {
          outs[o].index = o;
          outs[o].want_float = 0;
        }
        ret = rknn_outputs_get(ctx, io.n_output, outs, nullptr);
      }
      double post_ms = 0.0;
      records.clear();
      if (ret == 0) {
        const double p0 = t_run;
        postproc_predfl(outs, out_attrs, io.n_output, conf_thr, iou_thr, max_det);
        for (int i = 0; i < g_last_kept; ++i) {
          const candidate_t &candidate = g_candidates[i];
          records.push_back({candidate.x1, candidate.y1, candidate.x2, candidate.y2,
                             candidate.score, candidate.cls});
        }
        ret = rknn_outputs_release(ctx, io.n_output, outs);
        post_ms = now_ms() - p0;
      }
      resp.status = ret;
      resp.count = ret == 0 ? static_cast<std::uint32_t>(records.size()) : 0U;
      resp.run_ms = static_cast<float>(t_run - t0);
      resp.post_ms = static_cast<float>(post_ms);
      resp.candidates = g_last_candidates;
      resp.kept = g_last_kept;
      write_exact(std::cout, &resp, sizeof(resp));
      if (resp.status == 0 && !records.empty()) {
        write_exact(std::cout, records.data(), records.size() * sizeof(records[0]));
      }
      std::cout.flush();
    }

    if (ctx != 0) {
      rknn_destroy(ctx);
    }
    free(dummy);
    free_postproc_buffers();
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "det_worker error: " << exc.what() << "\n";
    return 1;
  }
}
