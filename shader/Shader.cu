#include <Global/Shader.cuh>
using namespace project;

OPTIX_GLOBAL_PARAM GlobalParams params;

namespace project {
    //设置和读取光线载荷
    __device__ __forceinline__ void setPayload(const float4 & payload, const float4 & albedo, const float4 normal) {
        optixSetPayload_0(__float_as_uint(payload.x));
        optixSetPayload_1(__float_as_uint(payload.y));
        optixSetPayload_2(__float_as_uint(payload.z));
        optixSetPayload_3(__float_as_uint(payload.w));

        optixSetPayload_4(__float_as_uint(albedo.x));
        optixSetPayload_5(__float_as_uint(albedo.y));
        optixSetPayload_6(__float_as_uint(albedo.z));
        optixSetPayload_7(__float_as_uint(albedo.w));

        optixSetPayload_8(__float_as_uint(normal.x));
        optixSetPayload_9(__float_as_uint(normal.y));
        optixSetPayload_10(__float_as_uint(normal.z));
        optixSetPayload_11(__float_as_uint(normal.w));
    }
    __device__ __forceinline__ void getPayload(float4 & result, float4 & albedo, float4 & normal) {
        result = {
                __uint_as_float(optixGetPayload_0()),
                __uint_as_float(optixGetPayload_1()),
                __uint_as_float(optixGetPayload_2()),
                __uint_as_float(optixGetPayload_3())
        };
        albedo = {
                __uint_as_float(optixGetPayload_4()),
                __uint_as_float(optixGetPayload_5()),
                __uint_as_float(optixGetPayload_6()),
                __uint_as_float(optixGetPayload_7())
        };
        normal = {
                __uint_as_float(optixGetPayload_8()),
                __uint_as_float(optixGetPayload_9()),
                __uint_as_float(optixGetPayload_10()),
                __uint_as_float(optixGetPayload_11())
        };
    }

    //发射光线
    __device__ __forceinline__ void rayTrace(
            const float3 & rayOriginInWorld, const float3 & rayDirection,
            float tMin, float tMax, OptixTraversableHandle handle,
            float4 & payload, float4 & albedo, float4 & normal)
    {
        //将float重解释为无符号整数以传递至optixTrace
        unsigned int p0, p1, p2, p3;
        p0 = __float_as_uint(payload.x);
        p1 = __float_as_uint(payload.y);
        p2 = __float_as_uint(payload.z);
        p3 = __float_as_uint(payload.w);

        unsigned int p4, p5, p6, p7;      //albedo
        p4 = __float_as_uint(albedo.x);
        p5 = __float_as_uint(albedo.y);
        p6 = __float_as_uint(albedo.z);
        p7 = __float_as_uint(albedo.w);

        unsigned int p8, p9, p10, p11;    //normal
        p8 = __float_as_uint(normal.x);
        p9 = __float_as_uint(normal.y);
        p10 = __float_as_uint(normal.z);
        p11 = __float_as_uint(normal.w);

        optixTrace(handle, rayOriginInWorld, rayDirection, tMin, tMax,
                   0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
                   0, 0, 0,
                   p0, p1, p2, p3,
                   p4, p5, p6, p7,
                   p8, p9, p10, p11);

        //读取追踪结果载荷
        payload.x = __uint_as_float(p0);
        payload.y = __uint_as_float(p1);
        payload.z = __uint_as_float(p2);
        payload.w = __uint_as_float(p3);

        albedo.x = __uint_as_float(p4);
        albedo.y = __uint_as_float(p5);
        albedo.z = __uint_as_float(p6);
        albedo.w = __uint_as_float(p7);

        normal.x = __uint_as_float(p8);
        normal.y = __uint_as_float(p9);
        normal.z = __uint_as_float(p10);
        normal.w = __uint_as_float(p11);
    }

    __device__ void closesthitImpl(GeometryType geometryType, MaterialType materialType) {
        const uint3 idx = optixGetLaunchIndex();
        const uint3 dim = optixGetLaunchDimensions();
        const unsigned int tid = idx.y * dim.x + idx.x;

        //获取当前颜色和当前追踪深度，判断递归终止条件
        float4 payload, albedo, normal;
        getPayload(payload, albedo, normal);
        if (payload.w >= rayTraceDepth) {
            setPayload(
                    {0.0f, 0.0f, 0.0f, payload.w},
                    {}, {});
            return;
        }
        const auto hitParams = reinterpret_cast<HitGroupParams *>(optixGetSbtDataPointer());

        //获取碰撞信息（HitRecord）所有向量均为世界坐标
        const float t = optixGetRayTmax();
        const float3 rayOrigin = optixGetWorldRayOrigin();
        const float3 rayDirection  = optixGetWorldRayDirection();
        const float3 hitPoint = rayOrigin + t * rayDirection;

        //获取发生碰撞的几何体在输入数组中的下标
        const unsigned int primitiveIndex = optixGetPrimitiveIndex();

        //根据不同的几何类型使用不同的碰撞点法线获取方式
        float3 normalVector = {};
        switch (geometryType) {
            case GeometryType::SPHERE: {
                //获取球体球心和半径
                const float3 sphereCenter = hitParams->sphere.centers[primitiveIndex];
                const float sphereRadius = hitParams->sphere.radii[primitiveIndex];

                /*
                 * 计算碰撞点法线
                 * outwardNormal为球面向外的单位法向量，通过此向量和光线方向向量的点积符号判断光线撞击了球的内表面还是外表面
                 * 若点积小于0，则两向量夹角大于90度，两向量不同方向
                 */
                const float3 outwardNormal = (hitPoint - sphereCenter) / sphereRadius; //可以使用normalize，效率较低
                const bool hitFrontFace = dot(rayDirection, outwardNormal) < 0.0f;
                normalVector = hitFrontFace ? outwardNormal : -outwardNormal;
                break;
            }
            case GeometryType::TRIANGLE: {
                //获取对应三角形顶点法线
                //如果 n1, n2, n3 在对象空间，需要变换到世界空间
                const float3 n1 = hitParams->triangles.vertexNormals[primitiveIndex * 3 + 0];
                const float3 n2 = hitParams->triangles.vertexNormals[primitiveIndex * 3 + 1];
                const float3 n3 = hitParams->triangles.vertexNormals[primitiveIndex * 3 + 2];

                //获取交点uv值
                const float2 barycentric = optixGetTriangleBarycentrics();
                const float u = barycentric.x;
                const float v = barycentric.y;
                const float w = 1.0f - u - v;

                //交点法向量为三个顶点法向量的插值平滑
                const float3 _normal = w * n1 + u * n2 + v * n3;
                const bool hitFrontFace = dot(rayDirection, _normal) < 0.0f;
                normalVector = hitFrontFace ? _normal : -_normal;
                break;
            }
            default:
                setPayload(
                        {1.0f, 0.6f, 0.8f, payload.w}, //错误颜色
                        {}, {}
                );
                return;
        }

        //根据材质类型使用不同的反射光线计算方法
        float3 reflectDirection = {};
        float3 _albedo = {};

        switch (materialType) {
            case MaterialType::ROUGH: {
                reflectDirection = normalVector + randomSpaceVector(params.stateArray + tid, 1.0f);

                //若随机的反射方向和法向量相互抵消，则取消随机反射
                if (MathHelper::floatValueEquals(lengthSquared(reflectDirection), FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE)) {
                    reflectDirection = normalVector;
                }

                _albedo = hitParams->rough.albedo;
                break;
            }
            case MaterialType::METAL: {
                const float3 v = rayDirection;
                const float3 n = normalVector;
                reflectDirection = normalize(v - 2 * dot(v, n) * n);

                //应用反射扰动：在距离物体表面1单位处随机选取单位向量和反射向量相加，形成随机扰动
                if (hitParams->metal.fuzz > 0.0f) {
                    reflectDirection += hitParams->metal.fuzz * randomSpaceVector(params.stateArray + tid, 1.0f);
                }

                _albedo = hitParams->metal.albedo;
                break;
            }
            default:
                setPayload(
                        {1.0f, 0.6f, 0.8f, payload.w}, //错误颜色
                        {}, {}
                );
                return;
        }

        //确保反射方向可用，避免光线方向数值错误
        if (!isfinite(reflectDirection.x) || !isfinite(reflectDirection.y) || !isfinite(reflectDirection.z) ||
            lengthSquared(reflectDirection) <= FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE)
        {
            //尝试使用法线
            reflectDirection = normalVector;
            if (lengthSquared(reflectDirection) <= FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE ||
                !isfinite(reflectDirection.x) || !isfinite(reflectDirection.y) || !isfinite(reflectDirection.z))
            {
                //使用固定值
                reflectDirection = make_float3(0.0f, 0.0f, 1.0f);
            }
        }

        //在首次弹射时写入额外信息
        if (MathHelper::floatValueEquals(payload.w, 1.0f)) {
            albedo.x = _albedo.x;
            albedo.y = _albedo.y;
            albedo.z = _albedo.z;
            albedo.w = 1.0f;

            const float3 _normal = normalize(normalVector);
            normal.x = _normal.x;
            normal.y = _normal.y;
            normal.z = _normal.z;
            normal.w = 0.0f;
        }

        //递归追踪：以当前命中点为起点发射一条新的光线
        float4 result = {payload.x, payload.y, payload.z, payload.w + 1.0f};
        rayTrace(
                hitPoint, reflectDirection, FLOAT_ZERO_VALUE, FLOAT_INFINITY_VALUE,
                params.handle, result, albedo, normal);

        //应用材质基础颜色
        result.x *= _albedo.x;
        result.y *= _albedo.y;
        result.z *= _albedo.z;

        //将最终着色结果写回当前光线的载荷中
        setPayload(result, albedo, normal);
    }
}

//raygen
OPTIX_PROGRAM void __raygen__raygenProgram() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float2 ndc = make_float2(
            ((static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x)) * 2.0f - 1.0f,
            ((static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y)) * 2.0f - 1.0f
    );

    //从raygen sbt中获取相机属性
    const auto data = reinterpret_cast<RayGenParams *>(optixGetSbtDataPointer());
    const float3 U = data->cameraU;
    const float3 V = data->cameraV;
    const float3 W = data->cameraW;
    const float3 origin = data->cameraCenter;
    const float aspect = static_cast<float>(data->width) / static_cast<float>(data->height);
    const float3 direction = normalize(ndc.x * aspect * U + ndc.y * V + W);

    //发射光线
    float4 result = {0.0f, 0.0f, 0.0f, 1.0f}, albedo, normal;
    rayTrace(
            origin, direction, FLOAT_ZERO_VALUE, FLOAT_INFINITY_VALUE,
            params.handle, result, albedo, normal);

    //写入颜色和辅助数据
    data->colorBuffer[idx.y * dim.x + idx.x] = colorToFloat4(result);
    data->albedoBuffer[idx.y * dim.x + idx.x] = albedo;
    data->normalBuffer[idx.y * dim.x + idx.x] = normal;
}

//miss
OPTIX_PROGRAM void __miss__missProgram() {
    //从sbt中获取颜色
    const auto data = reinterpret_cast<MissParams *>(optixGetSbtDataPointer());
    const auto backgroundColor = data->backgroundColor;

    //添加到光线载荷
    float4 payload, albedo, normal;
    getPayload(payload, albedo, normal);
    setPayload(
            {backgroundColor.x, backgroundColor.y, backgroundColor.z, payload.w},
            {}, {});
}

/*
 * 创建新的 Hit 程序：不同的几何类型，不同的材质算法，不同的着色逻辑（需要不同的代码路径）
 * 什么时候创建新的 SBT 记录：场景中的每个几何实例的不同参数，相同代码，不同数据
 *
 * 此处一个hit程序对应一种几何体的一种材质
 * 若要为一种几何体的一种材质创建多个不同的基础颜色，则创建多个SBT记录
 */
//hit - sphere
OPTIX_PROGRAM void __closesthit__closesthitProgramSphereRough() {
    closesthitImpl(GeometryType::SPHERE, MaterialType::ROUGH);
}
OPTIX_PROGRAM void __closesthit__closesthitProgramSphereMetal() {
    closesthitImpl(GeometryType::SPHERE, MaterialType::METAL);
}

//hit - triangle
OPTIX_PROGRAM void __closesthit__closesthitProgramTriangleRough() {
    closesthitImpl(GeometryType::TRIANGLE, MaterialType::ROUGH);
}
OPTIX_PROGRAM void __closesthit__closesthitProgramTriangleMetal() {
    closesthitImpl(GeometryType::TRIANGLE, MaterialType::METAL);
}
