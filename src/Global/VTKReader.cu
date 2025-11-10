#include <Global/VTKReader.cuh>
#include <JSON/json.hpp>
#include <fstream>
#include <sstream>
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
    std::tuple<std::vector<std::string>, std::vector<float>, size_t> VTKReader::readSeriesFile(const std::string & filePath) {
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

        std::vector<std::string> files;
        std::vector<float> times;

        for (const auto & item : data["files"]) {
            //从数组中的每个对象里提取 "name" 和 "time"
            const std::string name = item["name"];
            const float time = item["time"];
            files.push_back(name);
            times.push_back(time);
        }

        const size_t entryCount = files.size();
        const size_t printEntryCount = std::min(entryCount, static_cast<size_t>(5));
        SDL_Log("First %zd entries:", printEntryCount);
        for (size_t i = 0; i < printEntryCount; i++) {
            SDL_Log("Time: %f --> VTK file: %s", times[i], files[i].c_str());
        }

        SDL_Log("%s parse completed. Found %zd entries.", filePath.c_str(), entryCount);

        //time的输入为每个文件出现的时间，转换为每个文件持续的时间：后一个文件的出现时间减去当前文件出现时间
        std::vector<float> fileDurations(entryCount);
        if (entryCount == 1) {
            fileDurations[0] = 1000.0f;
        } else {
            for (size_t i = 0; i < entryCount - 1; i++) {
                fileDurations[i] = times[i + 1] - times[i];
            }
            //最后一个文件的出现时间使用倒数第二个文件的时间
            fileDurations[entryCount - 1] = fileDurations[entryCount - 2];
        }

        return {files, fileDurations, entryCount};
    }

    std::vector<VTKParticle> VTKReader::readVTKFile(const std::string & filePath) {
        //检查VTK文件头
        std::ifstream file(filePath);
        if (!file.is_open()) {
            SDL_Log("Failed to open vtk file: %s!", filePath.c_str());
            exit(-1);
        }
        std::string line;
        getline(file, line);
        if (line.find("# vtk DataFile Version") == std::string::npos) {
            SDL_Log("Illegal vtk file header in file %s: %s!", filePath.c_str(), line.c_str());
            exit(-1);
        }
        file.close();

        //读取VTK文件并获取vtkPolyData指针
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filePath.c_str());
        reader->Update();
        vtkPolyData * polyData = reader->GetOutput();
        if (polyData == nullptr || polyData->GetNumberOfPoints() == 0) {
            SDL_Log("Failed to get poly data pointer or there is no points in file %s!", filePath.c_str());
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
            SDL_Log("Failed to read cell data, in file %s!", filePath.c_str());
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
                SDL_Log("There is no points in cell %zd, in file %s!", i, filePath.c_str());
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
        std::vector<Particle> rendererParticles(particles.size());

        for (size_t i = 0; i < particles.size(); i++) {
            const auto & particle = particles[i];

            //vtkTriangleStrip由N个点组成N - 2个三角形
            std::vector<Triangle> triangles;
            const size_t triangleCount = particle.vertices.size() - 2;
            triangles.reserve(triangles.size() + triangleCount);

            //获取组成粒子的所有点，构造三角形数组
            for (size_t j = 0; j < triangleCount; j++) {
                std::array<float3, 3> vertices;
                std::array<float3, 3> normals;

                if ((j & 1) == 0) {
                    //偶数三角形，顶点顺序保持不变
                    vertices = {particle.vertices[j], particle.vertices[j + 1], particle.vertices[j + 2],};
                    normals = {particle.verticesNormals[j], particle.verticesNormals[j + 1], particle.verticesNormals[j + 2],};
                } else {
                    //奇数三角形，第2，3个顶点需要取反以保持面法线方向一致
                    vertices = {particle.vertices[j], particle.vertices[j + 2], particle.vertices[j + 1],};
                    normals = {particle.verticesNormals[j], particle.verticesNormals[j + 2], particle.verticesNormals[j + 1],};
                }

                //将三角形存入数组。粒子的材质单独存储，无需为每个三角形赋值
                const Triangle item = {
                        .vertices = vertices,
                        .normals = normals
                };
                triangles.push_back(item);
            }

            //将当前粒子的所有三角形构建粒子结构体
            rendererParticles[i].materialType = MaterialType::ROUGH;
            rendererParticles[i].materialIndex = 1;
            rendererParticles[i].triangles = triangles;
            rendererParticles[i].velocity = particle.velocity;
        }
        return rendererParticles;
    }

    void VTKReader::readSeriesFileToCache(const std::string & seriesFilePath, const std::string & seriesFileName, const std::string & cacheFilePath, bool isBinaryMode) {
        SDL_Log("Generating VTK cache...");

        //读取series文件
        const auto [files, times, fileCount] = readSeriesFile(seriesFilePath + seriesFileName);

        //创建缓存文件
        std::ofstream out;
        if (isBinaryMode) {
            out.open(cacheFilePath, std::ios::out | std::ios::binary);
        } else {
            out.open(cacheFilePath, std::ios::out);
        }

        if (!out) {
            SDL_Log("Error: Could not create or open cache file: %s!", cacheFilePath.c_str());
            exit(-1);
        }

        //读取所有VTK文件，并将返回值写入缓存文件
        for (size_t i = 0; i < fileCount; i++) {
            SDL_Log("[%zd/%zd] (%zd%%) Reading VTK file: %s...", i, fileCount,
                    static_cast<size_t>(static_cast<float>(i) / fileCount * 100.0f), files[i].c_str());
            const auto particles = readVTKFile(seriesFilePath + files[i]);

            if (isBinaryMode) {
                //二进制模式：直接写入原始数据
                size_t particleCount = particles.size();
                out.write(reinterpret_cast<const char*>(&particleCount), sizeof(size_t));

                for (const auto & particle: particles) {
                    //写入基础信息
                    out.write(reinterpret_cast<const char*>(&i), sizeof(size_t));
                    out.write(reinterpret_cast<const char*>(&particle.id), sizeof(particle.id));
                    out.write(reinterpret_cast<const char*>(&particle.velocity), sizeof(particle.velocity));
                    out.write(reinterpret_cast<const char*>(&particle.boundingBoxRanges), sizeof(particle.boundingBoxRanges));
                    out.write(reinterpret_cast<const char*>(&particle.centroid), sizeof(particle.centroid));

                    //写入顶点坐标
                    size_t vertexCount = particle.vertices.size();
                    out.write(reinterpret_cast<const char*>(&vertexCount), sizeof(size_t));
                    out.write(reinterpret_cast<const char*>(particle.vertices.data()), vertexCount * sizeof(float3));

                    //写入顶点法线
                    size_t normalCount = particle.verticesNormals.size();
                    out.write(reinterpret_cast<const char*>(&normalCount), sizeof(size_t));
                    out.write(reinterpret_cast<const char*>(particle.verticesNormals.data()), normalCount * sizeof(float3));
                }
            } else {
                //文本模式：转换为字符串后写入
                for (const auto & particle: particles) {
                    //第一行：基础信息
                    out << "[" << i << "]; "
                        << "ID = " << particle.id << "; "
                        << "Velocity = (" << particle.velocity.x << ", " << particle.velocity.y << ", " << particle.velocity.z << "); "
                        << "BoundingBox = [0](" << particle.boundingBoxRanges[0].x << ", " << particle.boundingBoxRanges[0].y << "), "
                        << "[1](" << particle.boundingBoxRanges[1].x << ", " << particle.boundingBoxRanges[1].y << "), "
                        << "[2](" << particle.boundingBoxRanges[2].x << ", " << particle.boundingBoxRanges[2].y << "); "
                        << "Centroid = (" << particle.centroid.x << ", " << particle.centroid.y << ", " << particle.centroid.z << ")\n";

                    //第二行：顶点坐标，保留原格式
                    for (const auto & vertex: particle.vertices) {
                        out << vertex.x << ", " << vertex.y << ", " << vertex.z << ", ";
                    }
                    out << "\n";

                    //第三行：顶点法线，保留原格式
                    for (const auto & normal: particle.verticesNormals) {
                        out << normal.x << ", " << normal.y << ", " << normal.z << ", ";
                    }
                    out << "\n";
                }
            }
        }
        out.close();

        SDL_Log("Cache generated, quit.");
        exit(0);
    }

    std::vector<std::vector<VTKParticle>> VTKReader::readVTKFromCache(const std::string & cacheFilePath, bool isBinaryMode) {
        SDL_Log("Reading VTK cache...");

        //打开缓存文件
        std::ifstream in;
        if (isBinaryMode) {
            in.open(cacheFilePath, std::ios::in | std::ios::binary);
        } else {
            in.open(cacheFilePath, std::ios::in);
        }

        if (!in) {
            SDL_Log("Error: Could not open cache file: %s!", cacheFilePath.c_str());
            exit(-1);
        }

        //用于存储所有帧的粒子数据
        std::vector<std::vector<VTKParticle>> allFrames;

        if (isBinaryMode) {
            //二进制模式：直接读取原始数据
            while (in.peek() != EOF) {
                size_t particleCount;
                in.read(reinterpret_cast<char*>(&particleCount), sizeof(size_t));
                if (in.eof()) break;

                for (size_t p = 0; p < particleCount; p++) {
                    VTKParticle particle;
                    size_t frameIndex;

                    //读取基础信息
                    in.read(reinterpret_cast<char*>(&frameIndex), sizeof(size_t));
                    in.read(reinterpret_cast<char*>(&particle.id), sizeof(particle.id));
                    in.read(reinterpret_cast<char*>(&particle.velocity), sizeof(particle.velocity));
                    in.read(reinterpret_cast<char*>(&particle.boundingBoxRanges), sizeof(particle.boundingBoxRanges));
                    in.read(reinterpret_cast<char*>(&particle.centroid), sizeof(particle.centroid));

                    //读取顶点坐标
                    size_t vertexCount;
                    in.read(reinterpret_cast<char*>(&vertexCount), sizeof(size_t));
                    particle.vertices.resize(vertexCount);
                    in.read(reinterpret_cast<char*>(particle.vertices.data()), vertexCount * sizeof(float3));

                    //读取顶点法线
                    size_t normalCount;
                    in.read(reinterpret_cast<char*>(&normalCount), sizeof(size_t));
                    particle.verticesNormals.resize(normalCount);
                    in.read(reinterpret_cast<char*>(particle.verticesNormals.data()), normalCount * sizeof(float3));

                    //确保allFrames有足够的空间存储当前帧
                    if (frameIndex >= allFrames.size()) {
                        allFrames.resize(frameIndex + 1);
                    }

                    //将粒子添加到对应帧
                    allFrames[frameIndex].push_back(particle);
                }
            }
        } else {
            //文本模式：解析字符串
            std::string line;
            while (std::getline(in, line)) {
                //第一行：基础信息
                VTKParticle particle;
                size_t frameIndex;

                //解析基础信息
                std::istringstream iss(line);
                char discard;
                iss >> discard >> frameIndex >> discard >> discard; //读取 "[frameIndex]; "

                std::string temp;
                iss >> temp >> discard >> particle.id >> discard; //读取 "ID = particle.id; "

                iss >> temp >> discard >> discard; //读取 "Velocity = ("
                iss >> particle.velocity.x >> discard >> particle.velocity.y >> discard >> particle.velocity.z >> discard >> discard; // 读取速度

                iss >> temp >> discard >> discard >> discard; //读取 "BoundingBox = [0]("
                iss >> particle.boundingBoxRanges[0].x >> discard >> particle.boundingBoxRanges[0].y >> discard >> discard; // 读取 boundingBox[0]
                iss >> discard >> discard >> discard; //读取 "[1]("
                iss >> particle.boundingBoxRanges[1].x >> discard >> particle.boundingBoxRanges[1].y >> discard >> discard; // 读取 boundingBox[1]
                iss >> discard >> discard >> discard; //读取 "[2]("
                iss >> particle.boundingBoxRanges[2].x >> discard >> particle.boundingBoxRanges[2].y >> discard >> discard; // 读取 boundingBox[2]

                iss >> temp >> discard >> discard; //读取 "Centroid = ("
                iss >> particle.centroid.x >> discard >> particle.centroid.y >> discard >> particle.centroid.z >> discard; // 读取质心

                //第二行：顶点坐标
                if (!std::getline(in, line)) {
                    SDL_Log("Error: Unexpected end of file while reading vertices!");
                    exit(-1);
                }
                std::istringstream vertexStream(line);
                while (vertexStream.good()) {
                    float3 vertex;
                    vertexStream >> vertex.x >> discard >> vertex.y >> discard >> vertex.z >> discard;
                    if (vertexStream.fail()) break;
                    particle.vertices.push_back(vertex);
                }

                //第三行：顶点法线
                if (!std::getline(in, line)) {
                    SDL_Log("Error: Unexpected end of file while reading normals!");
                    exit(-1);
                }
                std::istringstream normalStream(line);
                while (normalStream.good()) {
                    float3 normal;
                    normalStream >> normal.x >> discard >> normal.y >> discard >> normal.z >> discard;
                    if (normalStream.fail()) break;
                    particle.verticesNormals.push_back(normal);
                }

                //确保allFrames有足够的空间存储当前帧
                if (frameIndex >= allFrames.size()) {
                    allFrames.resize(frameIndex + 1);
                }

                //将粒子添加到对应帧
                allFrames[frameIndex].push_back(particle);
            }
        }
        in.close();

        SDL_Log("Cache read successfully, total frames: %zd", allFrames.size());
        return allFrames;
    }
}