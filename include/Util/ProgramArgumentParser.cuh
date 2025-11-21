#ifndef RENDEREROPTIX_PROGRAMARGUMENTPARSER_CUH
#define RENDEREROPTIX_PROGRAMARGUMENTPARSER_CUH

#include <Global/Renderer.cuh>
#include <JSON/json.hpp>
using json = nlohmann::json;

#define COMMAND_PARSER_ERROR_EXIT_CODE (-2)

/*
 * series-path：  series文件路径，可传入绝对路径或相对于可执行文件目录的相对路径
 * series-name：  series文件名，包含.series后缀
 * cache-path：   生成的缓存文件存放路径，可传入绝对路径或相对于可执行文件的相对路径
 * cache：        是否生成缓存文件并退出程序
 * debug-mode：   是否启用OptiX和图形API的调试模式
 *
 * roughs/metals: 额外几何体材质
 * spheres/triangles: 额外几何体
 *
 * api：使用图形API：OGL/VK/D3D11/D3D12
 * windowWidth/height/title：窗口大小和标题
 * fps：最大fps
 * camera-center/target：初始时相机位置和看向的位置
 * up-direction：相机上方方向向量，可为非单位向量
 * camera-pitch-limit-degree：相机俯仰角度限制，应小于90
 * camera-speed-stride：滚轮调节相机速度的变化量
 * camera-initial-speed-ratio：相机初始速度相对于camera-speed-stride的倍率
 * mouse-sensitivity：鼠标灵敏度
 * render-speed-ratio：粒子运动速度相对于原始速度的倍率，越大则越慢，1为原速
 * particle-shift/scale：所有粒子相对于VTK文件中位置的位移和缩放
 * cache-process-thread-count：读写缓存文件时使用的CPU线程数
 */
namespace project {
    class ProgramArgumentParser {
    public:
        //JSON配置文件路径
        static constexpr const char * CONFIG_FILE_PATH = "../files/config.json";

        //解析spheres数组
        static std::vector<std::array<float, 12>> parseSphereData(
                GeometryData & geoData, const json & sphereData);

        //解析config.json
        static std::tuple<
                GeometryData, MaterialData, Renderer::RenderLoopData,
                std::vector<std::array<float, 12>>,
                std::string, std::string, std::string, bool, bool, size_t
        > parseProgramArguments();
    };
}

#endif //RENDEREROPTIX_PROGRAMARGUMENTPARSER_CUH
