#ifndef RENDEREROPTIX_PROGRAMARGUMENTPARSER_CUH
#define RENDEREROPTIX_PROGRAMARGUMENTPARSER_CUH

#include <Global/HostFunctions.cuh>
#include <fstream>
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
        static std::vector<std::array<float, 12>> parseSphereData(GeometryData & geoData, const json & sphereData) {
            //额外几何体当前使用静态的变换矩阵，在此处预计算
            std::vector<std::array<float, 12>> transforms(sphereData.size());

            for (size_t i = 0; i < sphereData.size(); i++) {
                const auto & sphere = sphereData[i];

                //逐个对象成员解析
                const auto center = sphere["center"].get<std::vector<float>>();
                const auto radius = sphere["radius"].get<float>();
                const auto matType = sphere["mat-type"].get<std::string>() == "ROUGH"
                                     ? MaterialType::ROUGH : MaterialType::METAL;
                const auto matIndex = sphere["mat-index"].get<size_t>();
                const auto shift = sphere["shift"].get<std::vector<float>>();
                const auto rotate = sphere["rotate"].get<std::vector<float>>();
                const auto scale = sphere["scale"].get<std::vector<float>>();

                geoData.spheres.push_back({
                    {{center[0], center[1], center[2]}, radius}
                });
                geoData.sphereMaterialIndices.push_back({
                    matType, matIndex
                });
                //计算变换矩阵
                float transform[12];
                MathHelper::constructTransformMatrix(
                        {shift[0], shift[1], shift[2]},
                        {rotate[0], rotate[1], rotate[2]},
                        {scale[0], scale[1], scale[2]}, transform);
                //拷贝到数组
                memcpy(transforms[i].data(), transform, 12 * sizeof(float));
            }
            return transforms;
        }

        static std::tuple<
                GeometryData, MaterialData, Renderer::RenderLoopData,
                std::vector<std::array<float, 12>>,
                std::string, std::string, std::string, bool, bool, size_t
        > parseProgramArguments() {
            SDL_Log("Parsing program arguments from JSON file: %s", CONFIG_FILE_PATH);

            //打开 JSON 文件
            std::ifstream file(CONFIG_FILE_PATH);
            if (!file) {
                SDL_Log("Failed to open config: %s!", CONFIG_FILE_PATH);
                exit(COMMAND_PARSER_ERROR_EXIT_CODE);
            }

            //解析JSON数据
            json data;
            try {
                data = json::parse(file);

                //额外材质信息
                SDL_Log("Parsing additional material data...");
                MaterialData matData = {};
                for (const auto & rough: data["roughs"]) {
                    const auto albedo = rough["albedo"].get<std::vector<float>>();
                    matData.roughs.push_back({albedo[0], albedo[1], albedo[2]});
                }
                for (const auto & metal : data["metals"]) {
                    const auto albedo = metal["albedo"].get<std::vector<float>>();
                    const auto fuzz = metal["fuzz"].get<float>();
                    matData.metals.push_back({
                        {albedo[0], albedo[1], albedo[2]}, fuzz
                    });
                }

                //额外几何体信息
                SDL_Log("Parsing additional geometry data...");
                GeometryData geoData = {};
                const auto sphereTransforms = parseSphereData(geoData, data["spheres"]);

                //渲染窗口信息
                SDL_Log("Parsing render loop data...");

                const auto loopData = data["loop-data"];
                const std::string apiTypeStr = loopData["api"].get<std::string>();
                SDL_GraphicsWindowAPIType apiType;
                //平台检查：Linux不支持D3D11/D3D12
#ifndef _WIN32
                if (apiTypeStr == "D3D11" || apiTypeStr == "D3D12") {
                    SDL_Log("Error: Direct3D (D3D11/D3D12) is only supported on Windows!");
                    SDL_Log("Please use \"OGL\" or \"VK\" instead.");
                    exit(COMMAND_PARSER_ERROR_EXIT_CODE);
                }
#endif
                if (apiTypeStr == "OGL") {
                    apiType = SDL_GraphicsWindowAPIType::OPENGL;
                } else if (apiTypeStr == "VK") {
                    apiType = SDL_GraphicsWindowAPIType::VULKAN;
                } else if (apiTypeStr == "D3D11") {
                    apiType = SDL_GraphicsWindowAPIType::DIRECT3D11;
                } else if (apiTypeStr == "D3D12") {
                    apiType = SDL_GraphicsWindowAPIType::DIRECT3D12;
                } else {
                    SDL_Log("Invalid api type, must be \"OGL\", \"VK\", \"D3D11\" or \"D3D12\"!");
                    exit(COMMAND_PARSER_ERROR_EXIT_CODE);
                }
                const auto windowWidth = loopData["window-width"].get<int>();
                const auto windowHeight = loopData["window-height"].get<int>();
                const auto targetFPS = loopData["fps"].get<size_t>();
                const auto cameraCenterVec = loopData["camera-center"].get<std::vector<float>>();
                const auto cameraTargetVec = loopData["camera-target"].get<std::vector<float>>();
                const auto upDirectionVec = loopData["up-direction"].get<std::vector<float>>();
                const auto renderSpeedRatio = loopData["render-speed-ratio"].get<size_t>();
                const auto particleShiftVec = loopData["particle-shift"].get<std::vector<float>>();
                const auto particleScaleVec = loopData["particle-scale"].get<std::vector<float>>();
                const auto mouseSensitivity = loopData["mouse-sensitivity"].get<float>();
                const auto pitchLimitDegree = loopData["camera-pitch-limit-degree"].get<float>();
                const auto cameraSpeedStride = loopData["camera-speed-stride"].get<float>();
                const auto initialSpeedRatio = loopData["camera-initial-speed-ratio"].get<size_t>();
                const auto isDebugMode = data.at("debug-mode").get<bool>();

                const Renderer::RenderLoopData retLoopData(
                    apiType,
                    windowWidth, windowHeight, "RendererOptiX",
                    targetFPS,
                    {cameraCenterVec[0], cameraCenterVec[1], cameraCenterVec[2]},
                    {cameraTargetVec[0], cameraTargetVec[1], cameraTargetVec[2]},
                    {upDirectionVec[0], upDirectionVec[1], upDirectionVec[2]},
                    renderSpeedRatio,
                    {particleShiftVec[0], particleShiftVec[1], particleShiftVec[2]},
                    {particleScaleVec[0], particleScaleVec[1], particleScaleVec[2]},
                    mouseSensitivity,
                    pitchLimitDegree,
                    cameraSpeedStride,
                    initialSpeedRatio,
                    isDebugMode
                );
                file.close();

                return {
                    geoData, matData, retLoopData,
                    sphereTransforms,
                    data.at("series-path").get<std::string>(),
                    data.at("series-name").get<std::string>(),
                    data.at("cache-path").get<std::string>(),
                    data.at("cache").get<bool>(),
                    isDebugMode,
                    data.at("cache-process-thread-count").get<size_t>()
                };
            } catch (json::parse_error & e) {
                SDL_Log("JSON parsing error: %s", e.what());
                exit(COMMAND_PARSER_ERROR_EXIT_CODE);
            }
        }
    };
}

#endif //RENDEREROPTIX_PROGRAMARGUMENTPARSER_CUH
