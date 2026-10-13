// M1 harness, version 2: run small test shaders on the GPU.
//
// Usage:  m1_harness <chase|atomic> <path-to-shader.spv> [gpu_index]
//
//   chase  -> runs chase.spv       (follows a linked list by raw pointer)
//   atomic -> runs atomic_add.spv  (65,536 threads add 1.0 to one float)
//
// HOW TO READ THIS FILE
// The file is split into small functions, each doing ONE job. Start at main()
// at the bottom: it is only ~25 lines and calls the pieces in order:
//
//   create_context()  -> open a connection to the GPU
//   create_kernel()   -> hand the GPU our compiled shader
//   print_kernel_stats() -> ask the driver how "expensive" the shader is
//   test_chase() / test_atomic() -> prepare data, run the shader, check the answer
//
// Vulkan is verbose because it makes you describe everything explicitly.
// Most of the lines are "filling out a form" (the structs); the real work
// is the handful of vkXxx() calls.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ===========================================================================
// SECTION 1: small helpers
// ===========================================================================

// Almost every Vulkan function returns a VkResult. VK_SUCCESS means it worked.
// This macro stops the program with a readable message if anything else comes back.
#define VK_CHECK(call)                                                          \
  do {                                                                          \
    VkResult r_ = (call);                                                       \
    if (r_ != VK_SUCCESS) {                                                     \
      std::fprintf(stderr, "Vulkan error %d at %s:%d\n  %s\n", (int)r_,         \
                   __FILE__, __LINE__, #call);                                  \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

[[noreturn]] static void fail(const char *msg)
{
  std::fprintf(stderr, "FAIL: %s\n", msg);
  std::exit(1);
}

// The validation layer calls this whenever it spots a mistake in how we use Vulkan.
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                     VkDebugUtilsMessageTypeFlagsEXT,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                     void *)
{
  std::fprintf(stderr, "[validation] %s\n", data->pMessage);
  return VK_FALSE;
}

static std::vector<uint32_t> read_spirv(const char *path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    fail("could not open the .spv file (check the path)");
  const size_t bytes = (size_t)f.tellg();
  if (bytes == 0 || bytes % 4 != 0)
    fail(".spv file size is not a multiple of 4; it is not valid SPIR-V");
  std::vector<uint32_t> words(bytes / 4);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(words.data()), (std::streamsize)bytes);
  return words;
}

static bool device_has_extension(VkPhysicalDevice phys, const char *name)
{
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> exts(n);
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
  for (auto &e : exts)
    if (std::strcmp(e.extensionName, name) == 0)
      return true;
  return false;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t allowed_bits,
                                 VkMemoryPropertyFlags wanted)
{
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(phys, &props);
  for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
    const bool allowed = (allowed_bits & (1u << i)) != 0;
    const bool has_flags = (props.memoryTypes[i].propertyFlags & wanted) == wanted;
    if (allowed && has_flags)
      return i;
  }
  fail("no suitable memory type found");
}

// ===========================================================================
// SECTION 2: the "context" = our connection to one GPU
// ===========================================================================
struct Context {
  VkInstance instance = VK_NULL_HANDLE;           // connection to the Vulkan loader
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;         // the actual GPU (read-only info)
  VkDevice device = VK_NULL_HANDLE;               // our "session" with that GPU
  VkQueue queue = VK_NULL_HANDLE;                 // the line we submit work into
  uint32_t queue_family = 0;
  bool have_exec_props = false;                   // can we read register counts?
};

static Context create_context(int forced_gpu)
{
  Context ctx;

  // ---- Instance: ask for API 1.3 explicitly so we can't depend on 1.4 features
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "m1_harness";
  app.apiVersion = VK_API_VERSION_1_3;

  const char *validation_layer = "VK_LAYER_KHRONOS_validation";
  bool have_validation = false;
  {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> layers(n);
    vkEnumerateInstanceLayerProperties(&n, layers.data());
    for (auto &l : layers)
      if (std::strcmp(l.layerName, validation_layer) == 0)
        have_validation = true;
  }
  std::printf("Validation layer: %s\n", have_validation ? "ON" : "not found (continuing without)");

  std::vector<const char *> instance_exts;
  if (have_validation)
    instance_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledLayerCount = have_validation ? 1 : 0;
  ici.ppEnabledLayerNames = have_validation ? &validation_layer : nullptr;
  ici.enabledExtensionCount = (uint32_t)instance_exts.size();
  ici.ppEnabledExtensionNames = instance_exts.data();
  VK_CHECK(vkCreateInstance(&ici, nullptr, &ctx.instance));

  if (have_validation) {
    auto create_messenger = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        ctx.instance, "vkCreateDebugUtilsMessengerEXT");
    VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mci.pfnUserCallback = debug_callback;
    if (create_messenger)
      create_messenger(ctx.instance, &mci, nullptr, &ctx.messenger);
  }

  // ---- Pick a GPU
  uint32_t gpu_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &gpu_count, nullptr));
  if (gpu_count == 0)
    fail("no Vulkan GPUs found");
  std::vector<VkPhysicalDevice> gpus(gpu_count);
  VK_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &gpu_count, gpus.data()));

  int chosen = -1;
  std::printf("GPUs found:\n");
  for (uint32_t i = 0; i < gpu_count; i++) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpus[i], &p);
    std::printf("  [%u] %s (Vulkan %u.%u.%u, type %d)\n", i, p.deviceName,
                VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
                VK_VERSION_PATCH(p.apiVersion), (int)p.deviceType);
    if (chosen < 0 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      chosen = (int)i;
  }
  if (forced_gpu >= 0 && forced_gpu < (int)gpu_count)
    chosen = forced_gpu;
  if (chosen < 0)
    chosen = 0;
  ctx.phys = gpus[chosen];
  {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(ctx.phys, &p);
    std::printf("Using GPU [%d]: %s\n", chosen, p.deviceName);
  }

  // ---- Check features. Vulkan features are OFF until we explicitly switch them on.
  ctx.have_exec_props = device_has_extension(ctx.phys, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);

  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR have_exec{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 have2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  have2.pNext = &have12;
  if (ctx.have_exec_props)
    have12.pNext = &have_exec;
  vkGetPhysicalDeviceFeatures2(ctx.phys, &have2);

  if (!have12.bufferDeviceAddress)
    fail("GPU lacks bufferDeviceAddress");
  if (!have2.features.shaderInt64)
    fail("GPU lacks shaderInt64");
  if (ctx.have_exec_props && !have_exec.pipelineExecutableInfo)
    ctx.have_exec_props = false;

  // ---- Find a queue that can run compute
  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qf_count, qfs.data());
  ctx.queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      ctx.queue_family = i;
      break;
    }
  }
  if (ctx.queue_family == UINT32_MAX)
    fail("no compute queue family");

  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = ctx.queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  // ---- Create the logical device with only the features we need switched on
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR en_exec{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  en_exec.pipelineExecutableInfo = VK_TRUE;
  VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  en12.bufferDeviceAddress = VK_TRUE;
  if (ctx.have_exec_props)
    en12.pNext = &en_exec;
  VkPhysicalDeviceFeatures2 en2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  en2.pNext = &en12;
  en2.features.shaderInt64 = VK_TRUE;

  std::vector<const char *> device_exts;
  if (ctx.have_exec_props)
    device_exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &en2; // (with pNext features, pEnabledFeatures must stay null)
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)device_exts.size();
  dci.ppEnabledExtensionNames = device_exts.data();
  VK_CHECK(vkCreateDevice(ctx.phys, &dci, nullptr, &ctx.device));
  vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);

  std::printf("Register/statistics readout: %s\n",
              ctx.have_exec_props ? "available" : "not available on this driver");
  return ctx;
}

static void destroy_context(Context &ctx)
{
  vkDestroyDevice(ctx.device, nullptr);
  if (ctx.messenger != VK_NULL_HANDLE) {
    auto destroy_messenger = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        ctx.instance, "vkDestroyDebugUtilsMessengerEXT");
    if (destroy_messenger)
      destroy_messenger(ctx.instance, ctx.messenger, nullptr);
  }
  vkDestroyInstance(ctx.instance, nullptr);
  ctx = Context{};
}

// ===========================================================================
// SECTION 3: buffers that the shader can reach through raw pointers
// ===========================================================================
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceAddress address = 0; // the raw 64-bit GPU address the shader will use
  void *mapped = nullptr;      // CPU pointer to the same bytes
};

static Buffer create_buffer(const Context &ctx, VkDeviceSize size)
{
  Buffer b;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  // SHADER_DEVICE_ADDRESS is what lets us ask for the buffer's raw address.
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(ctx.device, &bci, nullptr, &b.buffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, b.buffer, &req);

  // The memory must ALSO be allocated with the device-address flag.
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &flags;
  mai.allocationSize = req.size;
  // HOST_VISIBLE + HOST_COHERENT: the CPU can read/write it directly.
  // Simple for a spike. Real rendering will use device-local memory + staging copies.
  mai.memoryTypeIndex = find_memory_type(
      ctx.phys, req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(ctx.device, &mai, nullptr, &b.memory));
  VK_CHECK(vkBindBufferMemory(ctx.device, b.buffer, b.memory, 0));
  VK_CHECK(vkMapMemory(ctx.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));

  VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  ai.buffer = b.buffer;
  b.address = vkGetBufferDeviceAddress(ctx.device, &ai);
  return b;
}

static void destroy_buffer(const Context &ctx, Buffer &b)
{
  vkUnmapMemory(ctx.device, b.memory);
  vkDestroyBuffer(ctx.device, b.buffer, nullptr);
  vkFreeMemory(ctx.device, b.memory, nullptr);
  b = Buffer{};
}

// ===========================================================================
// SECTION 4: a "kernel" = one compiled shader, ready to run
// ===========================================================================
// Both of our test shaders receive their inputs as a 16-byte push constant block.
static constexpr uint32_t kPushSize = 16;

struct Kernel {
  VkShaderModule shader = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

static Kernel create_kernel(const Context &ctx, const char *spv_path)
{
  Kernel k;

  std::vector<uint32_t> spirv = read_spirv(spv_path);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spirv.size() * sizeof(uint32_t);
  smci.pCode = spirv.data();
  VK_CHECK(vkCreateShaderModule(ctx.device, &smci, nullptr, &k.shader));

  // No descriptor sets: the shader gets everything through push constants
  // holding raw pointers.
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = kPushSize;

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &k.layout));

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = k.shader;
  cpci.stage.pName = "main"; // entry point name from the .slang file
  cpci.layout = k.layout;
  // Ask the driver to keep statistics (register counts etc.) about the compiled code.
  if (ctx.have_exec_props)
    cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  // This call is where the AMD driver compiles our SPIR-V into GCN machine code.
  VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &k.pipeline));
  std::printf("Pipeline created (driver compiled the shader).\n");
  return k;
}

static void destroy_kernel(const Context &ctx, Kernel &k)
{
  vkDestroyPipeline(ctx.device, k.pipeline, nullptr);
  vkDestroyPipelineLayout(ctx.device, k.layout, nullptr);
  vkDestroyShaderModule(ctx.device, k.shader, nullptr);
  k = Kernel{};
}

// Ask the driver what it learned while compiling. We print EVERY statistic it offers
// rather than guessing names, because drivers name them differently.
static void print_kernel_stats(const Context &ctx, const Kernel &k)
{
  if (!ctx.have_exec_props)
    return;
  auto get_execs = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(
      ctx.device, "vkGetPipelineExecutablePropertiesKHR");
  auto get_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(
      ctx.device, "vkGetPipelineExecutableStatisticsKHR");
  if (!get_execs || !get_stats)
    return;

  VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pi.pipeline = k.pipeline;
  uint32_t exec_count = 0;
  VK_CHECK(get_execs(ctx.device, &pi, &exec_count, nullptr));
  std::vector<VkPipelineExecutablePropertiesKHR> execs(exec_count);
  for (auto &e : execs) {
    e = {};
    e.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
  }
  VK_CHECK(get_execs(ctx.device, &pi, &exec_count, execs.data()));

  for (uint32_t i = 0; i < exec_count; i++) {
    std::printf("\nCompiled-shader statistics for \"%s\":\n", execs[i].name);
    VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = k.pipeline;
    ei.executableIndex = i;

    uint32_t stat_count = 0;
    VK_CHECK(get_stats(ctx.device, &ei, &stat_count, nullptr));
    std::vector<VkPipelineExecutableStatisticKHR> stats(stat_count);
    for (auto &s : stats) {
      s = {};
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    }
    VK_CHECK(get_stats(ctx.device, &ei, &stat_count, stats.data()));

    for (auto &s : stats) {
      switch (s.format) {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          std::printf("  %-32s %s\n", s.name, s.value.b32 ? "true" : "false");
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          std::printf("  %-32s %lld\n", s.name, (long long)s.value.i64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          std::printf("  %-32s %llu\n", s.name, (unsigned long long)s.value.u64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
          std::printf("  %-32s %g\n", s.name, s.value.f64);
          break;
        default:
          break;
      }
    }
  }
  std::printf("\n");
}

// Record one dispatch, submit it, wait for the GPU to finish. Returns elapsed milliseconds.
// `groups_x` = how many workgroups to launch. Each workgroup runs `numthreads` threads
// (that number is written inside the shader, e.g. [numthreads(64,1,1)]).
static double run_kernel(const Context &ctx, const Kernel &k, const void *push_data,
                         uint32_t groups_x)
{
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.queueFamilyIndex = ctx.queue_family;
  VkCommandPool pool;
  VK_CHECK(vkCreateCommandPool(ctx.device, &cpi, nullptr, &pool));

  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai, &cmd));

  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
  vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushSize, push_data);
  vkCmdDispatch(cmd, groups_x, 1, 1);
  VK_CHECK(vkEndCommandBuffer(cmd));

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &fence));

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  auto t0 = std::chrono::steady_clock::now();
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &si, fence));
  // 5 s timeout so a hung shader reports itself instead of freezing the program.
  VkResult wait = vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, 5'000'000'000ull);
  auto t1 = std::chrono::steady_clock::now();
  if (wait == VK_TIMEOUT)
    fail("GPU did not finish within 5 seconds (shader hang?)");
  VK_CHECK(wait);

  vkDestroyFence(ctx.device, fence, nullptr);
  vkDestroyCommandPool(ctx.device, pool, nullptr); // also frees the command buffer
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ===========================================================================
// SECTION 5: the two tests
// ===========================================================================

// ---- Test A: chase.spv ----------------------------------------------------
// Layouts here MUST match the Slang structs byte for byte (see chase.txt).
struct Node {
  uint32_t value;
  uint32_t padding; // 4 bytes of padding so `next` lands on an 8-byte boundary
  uint64_t next;    // GPU address of the next node, 0 = end of list
};
static_assert(sizeof(Node) == 16, "Node must be 16 bytes");
static_assert(offsetof(Node, next) == 8, "Node::next must be at offset 8");

struct ChasePush { // matches Params in chase.slang
  uint64_t head;   // offset 0
  uint64_t output; // offset 8
};
static_assert(sizeof(ChasePush) == kPushSize, "ChasePush must be 16 bytes");

static bool test_chase(const Context &ctx, const Kernel &k)
{
  const uint32_t N = 1000;
  Buffer nodes = create_buffer(ctx, N * sizeof(Node));
  Buffer output = create_buffer(ctx, sizeof(uint32_t));

  // Shuffle so the GPU really has to jump around memory by pointer.
  std::vector<uint32_t> order(N);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937 rng(42);
  std::shuffle(order.begin(), order.end(), rng);

  Node *slots = static_cast<Node *>(nodes.mapped);
  uint32_t expected_sum = 0;
  for (uint32_t i = 0; i < N; i++) {
    Node &n = slots[order[i]];
    n.value = i + 1;
    n.padding = 0;
    n.next = (i + 1 < N) ? nodes.address + (uint64_t)order[i + 1] * sizeof(Node) : 0;
    expected_sum += n.value;
  }
  *static_cast<uint32_t *>(output.mapped) = 0xDEADBEEF; // marker: "shader never ran"

  ChasePush pc{nodes.address + (uint64_t)order[0] * sizeof(Node), output.address};
  double ms = run_kernel(ctx, k, &pc, 1); // 1 workgroup of 1 thread

  const uint32_t gpu_sum = *static_cast<uint32_t *>(output.mapped);
  std::printf("CPU expected sum: %u\n", expected_sum);
  std::printf("GPU returned sum: %u (0x%08X)   [%.3f ms]\n", gpu_sum, gpu_sum, ms);

  destroy_buffer(ctx, output);
  destroy_buffer(ctx, nodes);
  return gpu_sum == expected_sum;
}

// ---- Test B: atomic_add.spv -----------------------------------------------
struct AtomicPush { // matches Params in atomic_add.slang
  uint64_t target;  // offset 0: address of the float everyone adds to
  float value;      // offset 8: what each thread adds
  uint32_t padding; // pad to 16 bytes
};
static_assert(sizeof(AtomicPush) == kPushSize, "AtomicPush must be 16 bytes");

static bool test_atomic(const Context &ctx, const Kernel &k)
{
  const uint32_t groups = 1024;          // workgroups launched
  const uint32_t threads_per_group = 64; // must equal numthreads in atomic_add.slang
  const uint32_t total_threads = groups * threads_per_group;

  Buffer target = create_buffer(ctx, sizeof(float));
  *static_cast<float *>(target.mapped) = 0.0f;

  AtomicPush pc{target.address, 1.0f, 0};
  double ms = run_kernel(ctx, k, &pc, groups);

  const float result = *static_cast<float *>(target.mapped);
  const float expected = (float)total_threads * pc.value; // 65536.0 is exact in float
  std::printf("Threads launched:  %u (each adds %.1f)\n", total_threads, pc.value);
  std::printf("CPU expected sum:  %.1f\n", expected);
  std::printf("GPU returned sum:  %.1f   [%.3f ms]\n", result, ms);

  destroy_buffer(ctx, target);
  return result == expected;
}

// ===========================================================================
// SECTION 6: main(), the whole program in 25 lines
// ===========================================================================
int main(int argc, char **argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <chase|atomic> <path-to-shader.spv> [gpu_index]\n", argv[0]);
    return 1;
  }
  const std::string mode = argv[1];
  const char *spv_path = argv[2];
  const int forced_gpu = (argc >= 4) ? std::atoi(argv[3]) : -1;
  if (mode != "chase" && mode != "atomic")
    fail("mode must be 'chase' or 'atomic'");

  Context ctx = create_context(forced_gpu);
  Kernel kernel = create_kernel(ctx, spv_path);
  print_kernel_stats(ctx, kernel);

  bool ok = (mode == "chase") ? test_chase(ctx, kernel) : test_atomic(ctx, kernel);
  std::printf("%s\n", ok ? "PASS" : "MISMATCH: see notes in the chat.");

  destroy_kernel(ctx, kernel);
  destroy_context(ctx);
  return ok ? 0 : 2;
}
