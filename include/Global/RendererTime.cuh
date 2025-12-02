#ifndef RENDEREROPTIX_RENDERERTIME_CUH
#define RENDEREROPTIX_RENDERERTIME_CUH

#include <Util/ColorRamp.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
#include <Global/RendererImpl.cuh>

/*
 * 非网格输入数据：所有粒子几何只构建一次，每帧根据VTK文件中每个粒子的速度和旋转四元数更新IAS
 * 需要周期性重建IAS以保持加速结构质量
 */
namespace project {
    typedef struct RendererTimeData {
        //额外几何体
        std::vector<RendererSphere> addSpheres;
        std::vector<RendererTriangle> addTriangles;

        //VTK粒子
        std::vector<RendererTimeParticle> vtkParticles;
        std::vector<float> durations;

        //VTK文件路径
        std::string seriesFilePath, seriesFileName;
        std::string cacheFilePath;

        //所有材质，材质数据直接存储
        RendererMaterial materialAllFiles;

        //加速结构
        RendererAS as;

        //OptiX变量
        OptixDeviceContext context;
        std::array<OptixModule, 3> modules;
        std::array<OptixProgramGroup, 6> programGroups;
        OptixPipeline pipeline;
        OptixDenoiser denoiser;

        //raygen和miss记录
        std::pair<CUdeviceptr, CUdeviceptr> raygenMissPtr;
        //着色器绑定表
        std::vector<SBT> sbt;

        //额外几何体实例更新函数
        UpdateAddInstancesFunc func;
    } RendererTimeData;

    class RendererTime {
    public:
        //初始化渲染器
        static RendererTimeData commitRendererData(
                GeometryData & addGeoData, MaterialData & addMatData,
                const std::string & seriesFilePath, const std::string & seriesFileName,
                bool isDebugMode);

        //设置额外实例更新函数
        static void setAddGeoInsUpdateFunc(RendererTimeData & data, UpdateAddInstancesFunc func);
    };
}

#endif //RENDEREROPTIX_RENDERERTIME_CUH
