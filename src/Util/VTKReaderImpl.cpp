//VTK
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkCell.h>
#include <vtkCellTypes.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>

//不能包含VTKReader.cuh或其他任何项目头文件，否则会引入OptiX头文件导致编译失败
//包含必要的环境头文件，这些头文件可以被GCC所识别
#include <cuda_runtime.h>
#include <SDL2/SDL.h>

#define VTK_READER_ERROR_EXIT_CODE (-1)
namespace vtk_reader {
    std::tuple<
            std::vector<size_t>, std::vector<float3>,
            std::vector<std::vector<float3>>, std::vector<std::vector<float3>>
    > readVTKFile(const std::string & filePath, std::atomic<size_t> & maxCellCountSingleFile) {
        std::vector<size_t> ids;
        std::vector<float3> velocities;
        std::vector<std::vector<float3>> verticesThisFile, normalsThisFile;

        //获取vtkPolyData指针
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filePath.c_str());
        reader->Update();
        vtkPolyData * polyData = reader->GetOutput();
        if (polyData == nullptr || polyData->GetNumberOfPoints() == 0) {
            SDL_Log("Failed to get poly data pointer or there is no points in file!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        const vtkIdType cellCount = polyData->GetNumberOfCells();
        maxCellCountSingleFile = std::max<size_t>(maxCellCountSingleFile, cellCount);

        //读取几何数据
        vtkCellData * cellData = polyData->GetCellData();
        vtkDataArray * idArray = cellData ? cellData->GetArray("id") : nullptr;
        vtkDataArray * velArray = cellData ? cellData->GetArray("vel") : nullptr;
        if (cellData == nullptr || idArray == nullptr || velArray == nullptr) {
            SDL_Log("Failed to read cell data!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }

        //计算全局顶点法向量
        vtkNew<vtkPolyDataNormals> normalsFilter;
        normalsFilter->SetInputData(polyData);
        normalsFilter->SetComputePointNormals(true); //计算顶点法向量
        normalsFilter->SetComputeCellNormals(false); //不计算面法向量
        normalsFilter->SetSplitting(false);          //不要因为法线差异而分裂顶点
        normalsFilter->SetConsistency(true);         //尝试使所有法线方向一致
        normalsFilter->SetAutoOrientNormals(true);   //将法向量定向到外侧
        normalsFilter->Update();

        vtkDataArray * vtkDataNormals = normalsFilter->GetOutput()->GetPointData()->GetNormals();
        vtkPoints * meshPoints = polyData->GetPoints();

        //逐个Cell读取
        for (vtkIdType i = 0; i < cellCount; i++) {
            //获取Cell作为独立几何对象
            vtkCell * cell = polyData->GetCell(i);

            //检查几何类型是否为vtkTriangleStrip
            if (strcmp(vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()), "vtkTriangleStrip") != 0) {
                SDL_Log("Found illegal cell type: %s!", vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()));
                exit(VTK_READER_ERROR_EXIT_CODE);
            }

            //ID
            ids.push_back(static_cast<size_t>(idArray->GetTuple1(i)));

            //速度
            const double * vel = velArray->GetTuple3(i);
            velocities.push_back({
                                         static_cast<float>(vel[0]),
                                         static_cast<float>(vel[1]),
                                         static_cast<float>(vel[2]),
                                 });

            //读取该Cell的所有顶点坐标和法向量
            vtkIdList * pointIds = cell->GetPointIds();
            const vtkIdType cellPointCount = cell->GetNumberOfPoints();

            //一个 TriangleStrip 会生成cellPointCount - 2个三角形
            const size_t triangleCount = cellPointCount - 2;
            std::vector<float3> verticesThisCell, normalsThisCell;
            verticesThisCell.reserve(triangleCount * 3); normalsThisCell.reserve(triangleCount * 3);

            for (vtkIdType triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
                /*
                 * 根据三角形索引调整顶点顺序
                 * 偶数三角形：正常顺序 (v_i, v_{i+1}, v_{i+2})
                 * 奇数三角形：翻转顺序 (v_{i}, v_{i+2}, v_{i+1})
                 */
                std::array<vtkIdType, 3> trianglePointIndexes = {
                        pointIds->GetId(triangleIndex + 0),
                        pointIds->GetId(triangleIndex + 1),
                        pointIds->GetId(triangleIndex + 2)
                };
                if ((triangleIndex & 1) != 0) {
                    std::swap(trianglePointIndexes[1], trianglePointIndexes[2]);
                }

                //添加三个顶点和法线
                for (vtkIdType pointIdx: trianglePointIndexes) {
                    double coords[3];
                    meshPoints->GetPoint(pointIdx, coords);
                    verticesThisCell.push_back({
                                                       static_cast<float>(coords[0]),
                                                       static_cast<float>(coords[1]),
                                                       static_cast<float>(coords[2])
                                               });
                    double normal[3];
                    vtkDataNormals->GetTuple(pointIdx, normal);
                    normalsThisCell.push_back({
                                                      static_cast<float>(normal[0]),
                                                      static_cast<float>(normal[1]),
                                                      static_cast<float>(normal[2])
                                              });
                }
            }

            //将顶点数组和法线数组移动到文件全局数组
            verticesThisFile.push_back(std::move(verticesThisCell));
            normalsThisFile.push_back(std::move(normalsThisCell));
        }
        return {ids, velocities, verticesThisFile, normalsThisFile};
    }
}