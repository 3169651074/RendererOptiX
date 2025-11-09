#include <Global/VTKReader.cuh>
#include <JSON/json.hpp>
#include <fstream>
using json = nlohmann::json;

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkCell.h>
#include <vtkCellTypes.h>
#include <vtkVersion.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>

namespace project {
    std::vector<std::pair<std::string, float>> VTKReader::readSeriesFile(const std::string & filePath) {
        SDL_Log("Reading series file: %s", filePath.c_str());

        //打开 JSON 文件
        std::ifstream file(filePath);
        if (!file.is_open()) {
            SDL_Log("Could not open the series file!");
            exit(-1);
        }

        json data;
        try {
            //使用 parse() 函数将文件流解析成一个 json 对象
            data = json::parse(file);
        } catch (json::parse_error & e) {
            //如果文件内容不是有效的 JSON 格式，会抛出异常
            SDL_Log("JSON parsing error: %s", e.what());
            exit(-1);
        }

        //从 JSON 对象中提取数据
        const std::string version = data["file-series-version"];
        SDL_Log("Series file version: %s", version.c_str());

        if (!(data.contains("files") && data["files"].is_array())) {
            SDL_Log("Failed to parse files array in series file!");
            exit(-1);
        }

        std::vector<std::pair<std::string, float>> fileEntries;
        for (const auto & item : data["files"]) {
            //从数组中的每个对象里提取 "name" 和 "time"
            const std::string name = item["name"];
            const float time = item["time"];
            fileEntries.push_back({name, time});
        }

        const size_t entryCount = fileEntries.size();
        const size_t printEntryCount = std::min(entryCount, static_cast<size_t>(5));
        SDL_Log("Found %zd vtk entries.", entryCount);
        SDL_Log("First %zd entries:", printEntryCount);
        for (size_t i = 0; i < printEntryCount; i++) {
            SDL_Log("Time: %f --> VTK file: %s", fileEntries[i].second, fileEntries[i].first.c_str());
        }

        SDL_Log("%s parse completed. Found %zd entries.", filePath.c_str(), fileEntries.size());
        return fileEntries;
    }

    std::vector<VTKParticle> VTKReader::readVTKFile(const std::string & filePath) {
        //SDL_Log("VTK version: %s", vtkVersion::GetVTKVersion());
        SDL_Log("Reading vtk file: %s...", filePath.c_str());

        //检查VTK文件头
        std::ifstream file(filePath);
        if (!file.is_open()) {
            SDL_Log("Failed to open vtk file: %s!", filePath.c_str());
            exit(-1);
        }
        std::string line;
        getline(file, line);
        if (line.find("# vtk DataFile Version") == std::string::npos) {
            SDL_Log("Illegal vtk file header: %s. In file %s!", line.c_str(), filePath.c_str());
            exit(-1);
        }
        file.close();

        //读取VTK文件并获取vtkPolyData指针
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filePath.c_str());
        reader->Update();
        vtkPolyData * polyData = reader->GetOutput();
        if (polyData == nullptr || polyData->GetNumberOfPoints() == 0) {
            SDL_Log("Failed to get poly data pointer or there is no points in this file!");
            exit(-1);
        }

        //全局点数量和粒子数量
        const vtkIdType numCells = polyData->GetNumberOfCells();
        //SDL_Log("Total point count: %zd, cell count: %zd.", polyData->GetNumberOfPoints(), numCells);
        std::vector<VTKParticle> ret;
        ret.reserve(numCells);

        //读取几何数据
        vtkCellData * cellData = polyData->GetCellData();
        vtkDataArray * idArray = cellData ? cellData->GetArray("id") : nullptr;
        vtkDataArray * velArray = cellData ? cellData->GetArray("vel") : nullptr;
        if (cellData == nullptr || idArray == nullptr || velArray == nullptr) {
            SDL_Log("Failed to read cell data!");
            exit(-1);
        }

        //计算全局顶点法向量
        vtkNew<vtkPolyDataNormals> normalsFilter;
        normalsFilter->SetInputData(polyData);
        normalsFilter->SetComputePointNormals(true); // 计算顶点法向量
        normalsFilter->SetComputeCellNormals(false); // 不计算面法向量
        normalsFilter->SetSplitting(false);          // 不要因为法线差异而分裂顶点
        normalsFilter->SetConsistency(true);         // 尝试使所有法线方向一致
        normalsFilter->SetAutoOrientNormals(true);   // 将法向量定向到外侧
        normalsFilter->Update();

        vtkPolyData * resultWithNormals = normalsFilter->GetOutput();
        vtkDataArray * normals = resultWithNormals->GetPointData()->GetNormals();
        vtkPoints * meshPoints = polyData->GetPoints();

        //逐个Cell读取
        for (vtkIdType i = 0; i < numCells; i++) {
            VTKParticle particle{};

            //获取Cell作为独立几何对象
            vtkCell * cell = polyData->GetCell(i);

            //检查几何类型是否为vtkTriangleStrip
            if (strcmp(vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()), "vtkTriangleStrip") != 0) {
                SDL_Log("Found illegal cell type: %s, aborting!", vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()));
                exit(-1);
            }

            //ID
            particle.id = static_cast<size_t>(idArray->GetTuple1(i));

            //速度
            const double * vel = velArray->GetTuple3(i);
            particle.velocity = float3{static_cast<float>(vel[0]), static_cast<float>(vel[1]), static_cast<float>(vel[2])};

            //包围盒
            double bounds[6];
            cell->GetBounds(bounds);
            for (size_t j = 0; j < 3; j++) {
                particle.boundingBoxRanges[j] = float2{static_cast<float>(bounds[j * 2]), static_cast<float>(bounds[j * 2 + 1])};
            }

            //质心
            const vtkIdType numCellPoints = cell->GetNumberOfPoints();
            if (numCellPoints > 0) {
                double centroid[3] = {0.0, 0.0, 0.0};
                vtkPoints * points = cell->GetPoints();

                //累加所有顶点坐标
                for (vtkIdType j = 0; j < numCellPoints; j++) {
                    double point[3];
                    points->GetPoint(j, point);
                    centroid[0] += point[0];
                    centroid[1] += point[1];
                    centroid[2] += point[2];
                }

                //取平均值
                centroid[0] /= static_cast<double>(numCellPoints);
                centroid[1] /= static_cast<double>(numCellPoints);
                centroid[2] /= static_cast<double>(numCellPoints);

                particle.centroid = float3{
                        static_cast<float>(centroid[0]),
                        static_cast<float>(centroid[1]),
                        static_cast<float>(centroid[2])};
            } else {
                SDL_Log("There is no points in cell %zd!", i);
                exit(-1);
            }

            //读取该Cell的所有顶点坐标和法向量
            particle.vertices.reserve(numCellPoints);
            particle.verticesNormals.reserve(numCellPoints);

            vtkIdList * pointIds = cell->GetPointIds();
            for (vtkIdType j = 0; j < numCellPoints; j++) {
                const vtkIdType pointId = pointIds->GetId(j);

                //获取顶点坐标
                double coords[3];
                meshPoints->GetPoint(pointId, coords);
                particle.vertices.push_back(float3{
                        static_cast<float>(coords[0]),
                        static_cast<float>(coords[1]),
                        static_cast<float>(coords[2]),
                });

                //获取顶点法向量
                double normal[3];
                normals->GetTuple(pointId, normal);
                particle.verticesNormals.push_back(float3{
                        static_cast<float>(normal[0]),
                        static_cast<float>(normal[1]),
                        static_cast<float>(normal[2]),
                });
            }
            ret.push_back(particle);
        }

        //SDL_Log("VTK file %s read completed.", filePath.c_str());
        return ret;
    }

    std::vector<Particle> VTKReader::convertToRendererData(const std::vector<VTKParticle> & particles) {
        //一个粒子对应一个实例，对应一个二维数组元素
        std::vector<Particle> particleTriangles(particles.size());

        for (size_t i = 0; i < particles.size(); i++) {
            //vtkTriangleStrip由N个点组成N - 2个三角形
            std::vector<Triangle> triangles;
            const size_t triangleCount = particles[i].vertices.size() - 2;
            triangles.reserve(triangles.size() + triangleCount);

            //获取组成粒子的所有点，构造三角形数组
            for (size_t j = 0; j < triangleCount; j++) {
                std::array<float3, 3> vertices;
                std::array<float3, 3> normals;

                if ((j & 1) == 0) {
                    //偶数三角形，顶点顺序保持不变
                    vertices = {particles[i].vertices[j], particles[i].vertices[j + 1], particles[i].vertices[j + 2],};
                    normals = {particles[i].verticesNormals[j], particles[i].verticesNormals[j + 1], particles[i].verticesNormals[j + 2],};
                } else {
                    //奇数三角形，第2，3个顶点需要取反以保持面法线方向一致
                    vertices = {particles[i].vertices[j], particles[i].vertices[j + 2], particles[i].vertices[j + 1],};
                    normals = {particles[i].verticesNormals[j], particles[i].verticesNormals[j + 2], particles[i].verticesNormals[j + 1],};
                }

                //将三角形存入数组。粒子的材质单独存储，无需为每个三角形赋值
                const Triangle item = {
                        .vertices = vertices,
                        .normals = normals
                };
                triangles.push_back(item);
            }

            //将当前粒子的所有三角形构建粒子结构体
            particleTriangles[i].materialType = MaterialType::ROUGH;
            particleTriangles[i].materialIndex = 1;
            particleTriangles[i].triangles = triangles;
        }
        return particleTriangles;
    }
}