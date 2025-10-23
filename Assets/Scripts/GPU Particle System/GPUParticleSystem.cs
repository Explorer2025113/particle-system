using UnityEngine;

public class GPUParticleSystem : MonoBehaviour
{
    [Header("Settings")]
    public int emissionRate = 1000; // 合理的发射率
    public float minLifetime = 5.0f; // 合理的生命周期
    public float maxLifetime = 10.0f; // 合理的生命周期
    public float minInitialSpeed = 0.5f;
    public float maxInitialSpeed = 2.0f;
    public Vector3 emitterPosition = Vector3.zero;
    public Vector3 emitterDirection = Vector3.zero; // 随机方向
    public float emitterRadius = 1.0f;
    public bool affectedByGravity = false; // 不受重力影响

    [Header("References")]
    public ComputeShader computeShader;
    public Material particleMaterial;
    public Mesh particleMesh;
    
    private int kernelUpdateArgs;
    private int kernelEmit;
    private int kernelSimulate;

    private GraphicsBuffer particleBuffer;
    private GraphicsBuffer deadPoolBuffer;
    private GraphicsBuffer aliveIndicesBufferA;
    private GraphicsBuffer aliveIndicesBufferB;
    private GraphicsBuffer countersBuffer;
    private GraphicsBuffer indirectArgsBuffer;

    private int pingPongA = 1; // Corresponds to _Counters[1]
    private int pingPongB = 2; // Corresponds to _Counters[2]

    private const int THREAD_COUNT_1D = 256;

    void OnEnable()
    {
        InitializeBuffers();
        InitializeParticles();
        SetupCamera();
    }
        
    void SetupCamera()
    {
        if (Camera.main != null)
        {
            Camera.main.backgroundColor = Color.black;
            Camera.main.clearFlags = CameraClearFlags.SolidColor;
            
            // 调整摄像机位置以更好地观察螺旋效果
            Camera.main.transform.position = new Vector3(0, 6, -15);
            Camera.main.transform.rotation = Quaternion.Euler(15, 0, 0);
        }
    }

    void OnDisable()
    {
        ReleaseBuffers();
    }
    
    void InitializeBuffers()
    {
        particleBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(float) * 16);
        deadPoolBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint));
        aliveIndicesBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint));
        aliveIndicesBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint));
        countersBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, sizeof(uint));
        indirectArgsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.IndirectArguments, 1, 5 * sizeof(uint));
    }

    void InitializeParticles()
    {
        kernelUpdateArgs = computeShader.FindKernel("UpdateArgs");
        kernelEmit = computeShader.FindKernel("Emit");
        kernelSimulate = computeShader.FindKernel("Simulate");

        uint[] deadIndices = new uint[ParticleConstants.MAX_PARTICLES];
        for (uint i = 0; i < ParticleConstants.MAX_PARTICLES; i++)
        {
            deadIndices[i] = i;
        }
        deadPoolBuffer.SetData(deadIndices);

        uint[] counters = new uint[] { (uint)ParticleConstants.MAX_PARTICLES, 0, 0 }; // 死粒子池满，活粒子为0
        countersBuffer.SetData(counters);

        uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
        if (particleMesh != null)
        {
            args[0] = particleMesh.GetIndexCount(0);
        }
        indirectArgsBuffer.SetData(args);
    }

    void Update()
    {
        // 1. UPDATE ARGS PASS
        computeShader.SetBuffer(kernelUpdateArgs, "_Counters", countersBuffer);
        computeShader.SetBuffer(kernelUpdateArgs, "_IndirectArgs", indirectArgsBuffer);
        computeShader.SetInt("_PingPong_A", pingPongA);
        computeShader.SetInt("_PingPong_B", pingPongB);
        computeShader.Dispatch(kernelUpdateArgs, 1, 1, 1);

        // 2. EMIT PASS
        int emissionCount = Mathf.Min(emissionRate, 100); // 每帧发射合理数量
        computeShader.SetInt("_EmissionCount", emissionCount);
        computeShader.SetFloat("_MinLifetime", minLifetime);
        computeShader.SetFloat("_MaxLifetime", maxLifetime);
        computeShader.SetFloat("_MinInitialSpeed", minInitialSpeed);
        computeShader.SetFloat("_MaxInitialSpeed", maxInitialSpeed);
        computeShader.SetVector("_EmitterPosition", emitterPosition);
        computeShader.SetVector("_EmitterDirection", emitterDirection);
        computeShader.SetFloat("_EmitterRadius", emitterRadius);
        computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value));
        computeShader.SetBuffer(kernelEmit, "_Particles", particleBuffer);
        computeShader.SetBuffer(kernelEmit, "_DeadPool", deadPoolBuffer);
        computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBufferA);
        computeShader.SetBuffer(kernelEmit, "_Counters", countersBuffer);
        computeShader.SetInt("_PingPong_A", pingPongA);
        int emitThreadGroups = Mathf.CeilToInt((float)emissionCount / THREAD_COUNT_1D);
        if (emitThreadGroups > 0)
        {
            computeShader.Dispatch(kernelEmit, emitThreadGroups, 1, 1);
        }

        // 3. SIMULATE PASS
        uint[] counters = new uint[3];
        countersBuffer.GetData(counters);
        uint simulationCount = counters[pingPongA];

        computeShader.SetFloat("_DeltaTime", Time.deltaTime);
        computeShader.SetInt("_AffectedByGravity", affectedByGravity ? 1 : 0);
        computeShader.SetBuffer(kernelSimulate, "_Particles", particleBuffer);
        computeShader.SetBuffer(kernelSimulate, "_DeadPool", deadPoolBuffer);
        computeShader.SetBuffer(kernelSimulate, "_AliveIndices_A", aliveIndicesBufferA);
        computeShader.SetBuffer(kernelSimulate, "_AliveIndices_B", aliveIndicesBufferB);
        computeShader.SetBuffer(kernelSimulate, "_Counters", countersBuffer);
        computeShader.SetBuffer(kernelSimulate, "_IndirectArgs", indirectArgsBuffer);
        computeShader.SetInt("_PingPong_A", pingPongA);
        computeShader.SetInt("_PingPong_B", pingPongB);

        int simulateThreadGroups = Mathf.CeilToInt((float)simulationCount / THREAD_COUNT_1D);
        if (simulateThreadGroups > 0)
        {
            computeShader.Dispatch(kernelSimulate, simulateThreadGroups, 1, 1);
        }

        // 4. RENDER
        Bounds bounds = new Bounds(Vector3.zero, new Vector3(100, 100, 100));
        particleMaterial.SetBuffer("_Particles", particleBuffer);
        particleMaterial.SetBuffer("_AliveIndices", aliveIndicesBufferB); // Use B (the output) for rendering
        Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, bounds, indirectArgsBuffer);
        
        // 每60帧输出一次GPU运行状态
        if (Time.frameCount % 60 == 0)
        {
            uint[] finalCounters = new uint[3];
            countersBuffer.GetData(finalCounters);
            Debug.Log($"🔥 GPU粒子系统运行中 - 活粒子: {finalCounters[pingPongB]}, 死粒子: {finalCounters[0]}, 发射线程组: {emitThreadGroups}, 模拟线程组: {simulateThreadGroups}");
        }

        // 5. SWAP BUFFERS for next frame
        int temp = pingPongA;
        pingPongA = pingPongB;
        pingPongB = temp;
    }

    void ReleaseBuffers()
    {
        particleBuffer?.Release();
        deadPoolBuffer?.Release();
        aliveIndicesBufferA?.Release();
        aliveIndicesBufferB?.Release();
        countersBuffer?.Release();
        indirectArgsBuffer?.Release();
    }
}