#include <Util/ProgramArgumentParser.cuh>
using namespace project;

static std::vector<std::array<float, 12>> instanceTransforms;
static void updateInstancesTransforms(
        OptixInstance * pin_instances, size_t instanceCount, unsigned long long frameCount)
{
    memcpy(pin_instances[0].transform, instanceTransforms[0].data(), 12 * sizeof(float));
}

#undef main
int main(int argc, char * argv[]) {
#ifdef MESH
    //解析JSON参数
    auto [geoData, matData, loopData, transforms,
          seriesFilePath, seriesFileName, cacheFilePath,
          isWriteCache, isDebugMode,
          cacheProcessThreadCount] = ProgramArgumentParser::parseProgramArguments();
    const size_t maxCacheLoadThreadCount = std::max<size_t>(1, cacheProcessThreadCount);
    if (isWriteCache) {
        RendererMesh::writeCacheFilesAndExit(seriesFilePath, seriesFileName, cacheFilePath, maxCacheLoadThreadCount);
    }
    instanceTransforms = transforms;

    //提交几何体，粒子文件信息和材质数据
    auto data = RendererMesh::commitRendererData(
            geoData, matData,
            seriesFilePath, seriesFileName, cacheFilePath,
            isDebugMode, maxCacheLoadThreadCount);
    RendererMesh::setAddGeoInsUpdateFunc(data, &updateInstancesTransforms);

    //启动交互
    RendererMesh::startRender(data, loopData);

    //清理资源
    RendererMesh::freeRendererData(data);
    return 0;
#elifdef TIME
#else

#endif
}
