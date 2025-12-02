#include <Global/RendererTime.cuh>

namespace project {
    RendererTimeData RendererTime::commitRendererData(
            GeometryData & addGeoData, MaterialData & addMatData,
            const std::string & seriesFilePath, const std::string & seriesFileName,
            bool isDebugMode)
    {

    }

    void RendererTime::setAddGeoInsUpdateFunc(RendererTimeData & data, UpdateAddInstancesFunc func) {
        if (func == nullptr) {
            SDL_Log("Update instances function pointer is null, additional instance will never be updated!");
        } else {
            SDL_Log("Update instances function set.");
            data.func = func;
        }
    }
}