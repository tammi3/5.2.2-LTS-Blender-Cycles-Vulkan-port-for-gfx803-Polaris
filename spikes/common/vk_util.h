// vk_util.h : the reusable Vulkan plumbing, extracted from the M1 harness.
//
// Same code you already ran in m1-harness/main.cpp, with two changes:
//   * create_kernel() takes the push-constant size, instead of a fixed 16 bytes
//   * run_kernel() can launch a 2D grid of workgroups (needed for images)
//
// Read the M1 main.cpp comments if you want the explanation of each step again.
// m1-harness is left untouched on purpose: it still works and we keep it as it was.
#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define VK_CHECK(call)                                                          \
  do {                                                                          \
    VkResult r_ = (call);                                                       \
    if (r_ != VK_SUCCESS) {                                                     \
      std::fprintf(stderr, "Vulkan error %d at %s:%d\n  %s\n", (int)r_,         \
                   __FILE__, __LINE__, #call);                                  \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

[[noreturn]] inline void fail(const char *msg)
{
  std::fprintf(stderr, "FAIL: %s\n", msg);
  std::exit(1);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                        VkDebugUtilsMessageTypeFlagsEXT,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                        void *)
{
  std::fprintf(stderr, "[validation] %s\n", data->pMessage);
  return VK_FALSE;
}

inline std::vector<uint32_t> read_spirv(const char *path)
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

inline bool device_has_extension(VkPhysicalDevice phys, const char *name)
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

inline uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t allowed_bits, VkMemoryPropertyFlags wanted)
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
// Context = our connection to one GPU
// ===========================================================================
struct Context {
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  bool have_exec_props = false;
};

inline Context create_context(int forced_gpu = -1)
{
  Context ctx;

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "cycles_vulkan_spike";
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
    mci.pfnUserCallback = vk_debug_callback;
    if (create_messenger)
      create_messenger(ctx.instance, &mci, nullptr, &ctx.messenger);
  }

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
  dci.pNext = &en2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)device_exts.size();
  dci.ppEnabledExtensionNames = device_exts.data();
  VK_CHECK(vkCreateDevice(ctx.phys, &dci, nullptr, &ctx.device));
  vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);
  return ctx;
}

inline void destroy_context(Context &ctx)
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
// Buffers reachable from shaders by raw 64-bit address
// ===========================================================================
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceAddress address = 0; // raw GPU address handed to the shader
  void *mapped = nullptr;      // CPU pointer to the same bytes
  VkDeviceSize size = 0;       // requested size in bytes
};

inline Buffer create_buffer(const Context &ctx, VkDeviceSize size)
{
  Buffer b;
  b.size = size;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(ctx.device, &bci, nullptr, &b.buffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, b.buffer, &req);

  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &flags;
  mai.allocationSize = req.size;
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

inline void destroy_buffer(const Context &ctx, Buffer &b)
{
  vkUnmapMemory(ctx.device, b.memory);
  vkDestroyBuffer(ctx.device, b.buffer, nullptr);
  vkFreeMemory(ctx.device, b.memory, nullptr);
  b = Buffer{};
}

// ===========================================================================
// Kernel = one compiled shader, ready to run
// ===========================================================================
struct Kernel {
  VkShaderModule shader = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  uint32_t push_size = 0;
};

inline Kernel create_kernel(const Context &ctx, const char *spv_path, uint32_t push_size)
{
  Kernel k;
  k.push_size = push_size;

  std::vector<uint32_t> spirv = read_spirv(spv_path);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = spirv.size() * sizeof(uint32_t);
  smci.pCode = spirv.data();
  VK_CHECK(vkCreateShaderModule(ctx.device, &smci, nullptr, &k.shader));

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = push_size;

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &k.layout));

  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = k.shader;
  cpci.stage.pName = "main";
  cpci.layout = k.layout;
  if (ctx.have_exec_props)
    cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &k.pipeline));
  std::printf("Pipeline created (driver compiled the shader).\n");
  return k;
}

inline void destroy_kernel(const Context &ctx, Kernel &k)
{
  vkDestroyPipeline(ctx.device, k.pipeline, nullptr);
  vkDestroyPipelineLayout(ctx.device, k.layout, nullptr);
  vkDestroyShaderModule(ctx.device, k.shader, nullptr);
  k = Kernel{};
}

inline void print_kernel_stats(const Context &ctx, const Kernel &k)
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

// Record one dispatch of groups_x * groups_y workgroups, submit, wait. Returns milliseconds.
// `push_data` must point to exactly k.push_size bytes.
inline double run_kernel(const Context &ctx, const Kernel &k, const void *push_data,
                         uint32_t groups_x, uint32_t groups_y = 1)
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
  vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, k.push_size, push_data);
  vkCmdDispatch(cmd, groups_x, groups_y, 1);
  VK_CHECK(vkEndCommandBuffer(cmd));

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &fence));

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  auto t0 = std::chrono::steady_clock::now();
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &si, fence));
  VkResult wait = vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, 5'000'000'000ull);
  auto t1 = std::chrono::steady_clock::now();
  if (wait == VK_TIMEOUT)
    fail("GPU did not finish within 5 seconds (shader hang?)");
  VK_CHECK(wait);

  vkDestroyFence(ctx.device, fence, nullptr);
  vkDestroyCommandPool(ctx.device, pool, nullptr);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
