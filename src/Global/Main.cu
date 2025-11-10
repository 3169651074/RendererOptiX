#include <Global/VTKReader.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
using namespace project;

namespace {
    void updateInstancesTransforms(std::vector<OptixInstance> & instances, unsigned long long frameCount) {
        //设置球体变换矩阵
        float sphereTransform[12];
        MathHelper::constructTransformMatrix(
                {0.0f, 0.0f, -1010.0f},
                {0.0f, 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f}, sphereTransform);
        memcpy(instances[0].transform, sphereTransform, 12 * sizeof(float));
    }
}

#undef main

#define USE_GENERATED_VTK_CACHE
int main(int argc, char * argv[]) {
#ifdef GENERATE_VTK_CACHE
    VTKReader::readSeriesFileToCache("../files/", "particle_mesh.vtk.series", "../cache/VTKParticleCache.cache");
#endif

    //初始化optix
    auto context = createContext();

    /*
     * 在渲染前需要将所有VTK文件包含的几何体信息，连同额外添加的几何体一同加载到设备内存中以备随时访问
     * 则需要创建多组加速结构。一个文件一个IAS，一组实例和一组GAS
     * 渲染时需要保证可以取到需要的IAS，通过IAS自动索引实例列表和GAS列表
     *
     * 构造加速结构时，将每个文件的所有粒子和额外添加的几何实体共同构建例列表和IAS
     * 在加速结构二维数组中，根据文件编号索引IAS
     */
    std::vector<std::vector<GAS>> gasForAllFiles;
    std::vector<std::vector<OptixInstance>> instancesForAllFiles;
    std::vector<IAS> iasForAllFiles;

    //球体
    const std::vector<Sphere> spheres = {
            {MaterialType::ROUGH, 3, float3{0.0f, 0.0f, 0.0f}, 1000.0f},
    };
    auto sphereGAS = buildGASForSpheres(context, spheres);
    const size_t additionalGeometryCount = spheres.size(); //额外几何体的总数，用于偏移粒子的SBT下标

    //粒子
    std::vector<std::vector<Particle>> particlesForAllFiles;

    //读取series文件
    const std::string seriesFilePath("../files/");
    const auto [files, durations, fileCount] = VTKReader::readSeriesFile(seriesFilePath + "particle_mesh.vtk.series");

#ifndef USE_GENERATED_VTK_CACHE
    for (size_t i = 0; i < fileCount; i++) {
        //读取VTK粒子并转换为粒子数组
        SDL_Log("[%zd/%zd] (%zd%%) Reading VTK file: %s...", i, fileCount,
                static_cast<size_t>(static_cast<float>(i) / fileCount * 100.0f), files[i].c_str());
        const auto vtkParticles = VTKReader::readVTKFile(seriesFilePath + files[i]);
        const auto particlesForThisFile = VTKReader::convertToRendererData(vtkParticles);
        const auto particleCountForThisFile = particlesForThisFile.size();
        particlesForAllFiles.push_back(particlesForThisFile);

        //为粒子数组中每一个粒子构造GAS列表和实例列表
        std::vector<GAS> gasForThisFile;
        gasForThisFile.reserve(particleCountForThisFile);

        //添加独立几何体GAS，添加顺序需要和SBT记录创建顺序相同（球体 -> 三角形 -> 粒子）
        gasForThisFile.push_back(sphereGAS);

        //添加粒子GAS
        for (size_t j = 0; j < particleCountForThisFile; j++) {
            gasForThisFile.push_back(buildGASForTriangles(context, particlesForThisFile[j].triangles));
        }
        gasForAllFiles.push_back(gasForThisFile);

        const auto instancesForThisFile = createInstances(gasForThisFile);
        instancesForAllFiles.push_back(instancesForThisFile);

        //将当前文件所有粒子放到IAS中
        iasForAllFiles.push_back(buildIAS(context, instancesForThisFile));
    }
#else
    const auto vtkParticles = VTKReader::readVTKFromCache("../cache/VTKParticleCache.cache");

    //处理每个文件
    for (size_t i = 0; i < fileCount; i++) {
        //转换为粒子数组
        SDL_Log("[%zd/%zd] (%zd%%) Converting VTK file: %s...", i, fileCount,
                static_cast<size_t>(static_cast<float>(i) / fileCount * 100.0f), files[i].c_str());
        const auto particlesForThisFile = VTKReader::convertToRendererData(vtkParticles[i]);
        const auto particleCountForThisFile = particlesForThisFile.size();
        particlesForAllFiles.push_back(particlesForThisFile);

        //为粒子数组中每一个粒子构造GAS列表和实例列表
        std::vector<GAS> gasForThisFile;
        gasForThisFile.reserve(particleCountForThisFile);

        //添加独立几何体GAS，添加顺序需要和SBT记录创建顺序相同（球体 -> 三角形 -> 粒子）
        gasForThisFile.push_back(sphereGAS);

        //添加粒子GAS
        for (size_t j = 0; j < particleCountForThisFile; j++) {
            gasForThisFile.push_back(buildGASForTriangles(context, particlesForThisFile[j].triangles));
        }
        gasForAllFiles.push_back(gasForThisFile);

        const auto instancesForThisFile = createInstances(gasForThisFile);
        instancesForAllFiles.push_back(instancesForThisFile);

        //将当前文件所有粒子放到IAS中
        iasForAllFiles.push_back(buildIAS(context, instancesForThisFile));
    }
#endif

    //创建模块和管线，所有文件都相同
    const OptixPipelineCompileOptions pipelineCompileOptions = {
            .usesMotionBlur = 0,
            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
            .numPayloadValues = 4,      //3个颜色分量+1个当前追踪深度
            .numAttributeValues = 3,
            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
            .pipelineLaunchParamsVariableName = "params",
            .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    };
    auto modules = createModules(context, pipelineCompileOptions);
    auto programs = createProgramGroups(context, modules);
    auto pipeline = linkPipeline(context, programs, pipelineCompileOptions);

    //初始化窗口和全局资源，所有文件都相同
    const auto type = SDL_GraphicsWindowAPIType::OPENGL;
    const int w = 1200, h = 800;
    const float fps = 60.0f;
    auto camera = SDL_GraphicsWindowConfigureCamera(
            {5.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, type
    );
    auto args = SDL_CreateGraphicsWindow(
            "Test", w, h, type, fps);
    //初始化随机数生成器
    curandState * dev_stateArray = nullptr;
    RandomGenerator::initDeviceRandomGenerators(dev_stateArray, w, h);

    //设置全局参数
    GlobalParams params = {
            .handle = 0,
            .stateArray = dev_stateArray
    };
    CUdeviceptr dev_params = 0;
    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_params), sizeof(GlobalParams)));

    //当前使用的文件下标，即加速结构索引
    size_t currentFileIndex = 0;
    //一个文件渲染的帧数
    const size_t frameCountPerFile = static_cast<size_t>(0.02f * fps * 2.0f);

    //所有粒子的位置偏移（相对于原始位置）和缩放
    const float3 particleOffset = {0.0f, 0.0f, 1.0f};
    const float3 particleScale = {1.0f, 1.0f, 1.0f};

    //循环每个文件
    SDL_Event event;
    SDL_GraphicsWindowKeyMouseInput input;
    SDL_Log("Starting...");

    while (currentFileIndex < fileCount) {
        //准备当前文件资源
        auto & particlesForThisFile = particlesForAllFiles[currentFileIndex];
        auto & instancesForThisFile = instancesForAllFiles[currentFileIndex];
        auto & iasForThisFile = iasForAllFiles[currentFileIndex];

        //设置几何体输入和材质输入，创建着色器绑定表
        const GeometryData geoData = {
                .spheres = spheres,
                .particles = particlesForThisFile
        };
        const MaterialData matData = {
                .roughs = {
                        {.65, .05, .05},
                        {.73, .73, .73},
                        {.12, .45, .15},
                        {.70, .60, .50},
                },
                .metals = {
                        {0.8, 0.85, 0.88, 0.0},
                }
        };
        auto sbt = createShaderBindingTable(programs, geoData, matData);

        //启动渲染
        unsigned long long frameCount = 0;

        while (frameCount < frameCountPerFile) {
            SDL_GraphicsWindowFrameStart(args);
            SDL_GraphicsWindowUpdateCamera(event, input, args, camera);

            //更新粒子变换矩阵
            float transform[12];
            for (size_t i = additionalGeometryCount; i < instancesForThisFile.size(); i++) {
                //计算该粒子运动总位移向量
                const auto & velocity = particlesForThisFile[i - additionalGeometryCount].velocity;
                const float3 totalShift = velocity * durations[currentFileIndex];

                //获取当前帧运动的位移，加上偏移量得到当前帧的位置
                const float3 shift = totalShift / static_cast<float>(frameCountPerFile);
                MathHelper::constructTransformMatrix(
                        {particleOffset + shift * frameCount},
                        {0.0f, 0.0f, 0.0f}, particleScale, transform);
                memcpy(instancesForThisFile[i].transform, transform, 12 * sizeof(float));
            }
            //更新额外几何体变换矩阵
            updateInstancesTransforms(instancesForThisFile, frameCount);
            //更新IAS
            updateIAS(context, iasForThisFile, instancesForThisFile);

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
            params.handle = std::get<0>(iasForThisFile);
            cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_params),&params, sizeof(GlobalParams),cudaMemcpyHostToDevice));

            //启动
            optixCheckError(optixLaunch(
                    pipeline, nullptr, dev_params, sizeof(GlobalParams),
                    &sbt.first, w, h, 1));
            cudaCheckError(cudaDeviceSynchronize());

            //显示
            SDL_GraphicsWindowPresentFrame(args);
            SDL_GraphicsWindowFrameFinish(args);

            //更新帧计数
            frameCount++;
        }
        //清理此文件资源
        freeShaderBindingTable(sbt);

        //下一个文件
        if (input.keyQuit) break;
        currentFileIndex++;
    }

    //清理窗口资源和全局资源
    SDL_Log("Finished.");
    SDL_DestroyGraphicsWindow(args);
    cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_params)));
    RandomGenerator::freeDeviceRandomGenerators(dev_stateArray);

    //清理optix资源
    SDL_Log("Render finished, cleaning up resources...");
    unlinkPipeline(pipeline);
    destroyProgramGroups(programs);
    destroyModules(modules);

    //释放加速结构设备内存，球体GAS需要单独释放
    cleanupAccelerationStructure(iasForAllFiles);
    for (auto & gasForThisFile : gasForAllFiles) {
        for (size_t i = additionalGeometryCount; i < gasForThisFile.size(); i++) {
            cleanupAccelerationStructure(gasForThisFile[i]);
        }
    }
    cleanupAccelerationStructure(sphereGAS);

    destroyContext(context);
    SDL_Log("Cleanup completed.");
    return 0;
}
