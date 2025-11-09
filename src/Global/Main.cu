#include <Global/VTKReader.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
using namespace project;

namespace {
    void updateInstancesTransforms(OptixInstance * dev_instances, size_t instanceCount, size_t frameCount) {
        float transform[12];
        MathHelper::constructTransformMatrix(
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0}, transform);
        for (size_t i = 0; i < instanceCount; i++) {
            cudaCheckError(cudaMemcpy((dev_instances + i)->transform, transform, 12 * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

#undef main
int main(int argc, char * argv[]) {
    //初始化optix
    auto context = createContext();

    //读取VTK文件
    const auto vtkParticles = VTKReader::readVTKFile("../files/particle_mesh/particle_000000000040000.vtk");
    const size_t particleCount = vtkParticles.size();

    //将文件中的所有粒子信息转换为粒子数组
    const auto particles = VTKReader::convertToRendererData(vtkParticles);

    //为每个粒子构建一个GAS并将指针存入数组：GAS句柄，GAS设备指针，当前粒子的SBT记录下标（当前设置为0，使用同一个记录）
    std::vector<GAS> gasArray(particleCount);
    for (size_t i = 0; i < particles.size(); i++) {
        auto [handle, ptr] = buildGASForTriangles(context, particles[i].triangles);
        gasArray[i] = {handle, ptr}; //实例的sbtOffset属性决定此实例使用哪一个颜色
    }

    //构建实例列表
    auto instancePtr = createInstances(gasArray);
    //构建IAS
    auto ias = buildIAS(context, instancePtr, particleCount);
    auto & [iasHandle, _, _1] = ias;

    //创建模块和管线
    const OptixPipelineCompileOptions pipelineCompileOptions = {
            .usesMotionBlur = 0,
            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
            .numPayloadValues = 4,      //3个颜色分量+1个当前追踪深度
            .numAttributeValues = 3,
            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
            .pipelineLaunchParamsVariableName = "params",
            .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE)
    };
    auto modules = createModules(context, pipelineCompileOptions);
    auto programs = createProgramGroups(context, modules);
    auto pipeline = linkPipeline(context, programs, pipelineCompileOptions);

    //几何体输入和材质输入
    const GeometryData geoData = {
            .particles = particles
    };
    const MaterialData matData = {
            .roughs = {
                    {.65, .05, .05},
                    {.73, .73, .73},
                    {.12, .45, .15},
                    {.70, .60, .50}
            },
            .metals = {
                    {0.8, 0.85, 0.88, 0.0}
            }
    };
    //每个实例对应一条sbt记录
    auto sbt = createShaderBindingTable(programs, geoData, matData);

    //设置窗口
    const auto type = SDL_GraphicsWindowAPIType::OPENGL;
    const int w = 1200, h = 800;
    auto camera = SDL_GraphicsWindowConfigureCamera(
            {3.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, type
    );
    auto args = SDL_CreateGraphicsWindow(
            "Test", w, h, type, 60);

    //初始化随机数生成器
    curandState * dev_stateArray = nullptr;
    RandomGenerator::initDeviceRandomGenerators(dev_stateArray, w, h);

    //设置全局参数
    const GlobalParams params = {
            .handle = iasHandle,
            .stateArray = dev_stateArray
    };
    CUdeviceptr dev_params = 0;
    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_params), sizeof(GlobalParams)));

    //启动渲染
    SDL_Log("Starting...");
    SDL_Event event;
    SDL_GraphicsWindowKeyMouseInput input;
    size_t frameCount = 0;

    while (!input.keyQuit) {
        SDL_GraphicsWindowFrameStart(args);
        SDL_GraphicsWindowUpdateCamera(event, input, args, camera);

        //更新raygen
        const RayGenParams rgData = {
                .width = w,
                .height = h,
                .surfaceObject = SDL_GraphicsWindowPrepareFrame(args),
                .cameraCenter = camera.cameraCenter,
                .cameraU = camera.cameraU,
                .cameraV = camera.cameraV,
                .cameraW = camera.cameraW
        };
        cudaCheckError(cudaMemcpy(
                reinterpret_cast<void *>(sbt.second[0] + OPTIX_SBT_RECORD_HEADER_SIZE), //SBT记录的头部是optix的header字段
                &rgData, sizeof(RayGenParams), cudaMemcpyHostToDevice));

        //更新全局参数
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_params),&params, sizeof(GlobalParams),cudaMemcpyHostToDevice));

        //启动
        optixCheckError(optixLaunch(
                pipeline, nullptr, dev_params, sizeof(GlobalParams),
                &sbt.first, w, h, 1));
        cudaCheckError(cudaDeviceSynchronize());

        //显示
        SDL_GraphicsWindowPresentFrame(args);
        SDL_GraphicsWindowFrameFinish(args);

        //更新实例
        frameCount++;
        //updateInstancesTransforms(instancePtr, particleCount, frameCount);
        //updateIAS(context, ias, instancePtr, particleCount); //需要更新全局参数的handle
    }
    //清理窗口资源
    SDL_Log("Finished.");
    SDL_DestroyGraphicsWindow(args);
    cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_params)));
    RandomGenerator::freeDeviceRandomGenerators(dev_stateArray);

    //清理资源
    SDL_Log("Render finished, cleaning up resources...");
    freeShaderBindingTable(sbt);
    unlinkPipeline(pipeline);
    destroyProgramGroups(programs);
    destroyModules(modules);

    cleanupAccelerationStructure(ias);
    freeInstances(instancePtr);
    cleanupAccelerationStructure(gasArray);
    destroyContext(context);

    SDL_Log("Cleanup completed.");
    return 0;
}
