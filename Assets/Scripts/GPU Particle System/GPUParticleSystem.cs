using UnityEngine;

// (No changes needed at the top of the file)
public class GPUParticleSystem : MonoBehaviour
{
    public enum EmissionType { Random, Spiral }
    [Header("Emission Settings")]
    public EmissionType emissionType = EmissionType.Spiral;
    public int emissionRate = 100000;
    public float minLifetime = 5.0f;
    public float maxLifetime = 10.0f;
    public float minInitialSpeed = 0.5f;
    public float maxInitialSpeed = 2.0f;
    public Vector3 emitterPosition = Vector3.zero;
    public Vector3 emitterDirection = Vector3.up;
    public float emitterRadius = 1.0f;
    [Header("Spiral Settings")]
    public float spiralForce = 0.1f;
    public float spiralRadius = 2.0f;
    public float spiralHeight = 5.0f;
    public float spiralTurns = 3.0f;
    [Header("General Settings")]
    public bool prewarm = true;
    public float prewarmTime = 10.0f;
    public bool affectedByGravity = false;
    public Vector3 renderBounds = new Vector3(100, 100, 100);
    [Header("References")]
    public ComputeShader computeShader;
    public Material particleMaterial;
    public Mesh particleMesh;
    
    private int kernelUpdateArgs, kernelEmit, kernelSimulate;
    private GraphicsBuffer particleBuffer, deadPoolBuffer, aliveIndicesBufferA, aliveIndicesBufferB, countersBuffer, indirectArgsBuffer;
    private GraphicsBuffer[] aliveIndicesBuffers;
    private int pingPongA = 1, pingPongB = 2;
    private const int THREAD_COUNT_1D = 256;
    
    // OnEnable remains the same
    void OnEnable()
    {
        InitializeBuffers();
        InitializeParticles();
        SetupCamera();
        
        if (prewarm)
        {
            Debug.Log("🔥 Pre-warming particle system...");
            SetShaderParameters();
            float fixedDeltaTime = 1.0f / 30.0f;
            int prewarmSteps = Mathf.CeilToInt(prewarmTime / fixedDeltaTime);
            for (int i = 0; i < prewarmSteps; i++)
            {
                RunSimulationStep(fixedDeltaTime);
            }
            Debug.Log("🔥 Pre-warm complete.");
        }
    }

    // OnDisable remains the same
    void OnDisable()
    {
        ReleaseBuffers();
    }

    void Update()
    {
        SetShaderParameters();
        RunSimulationStep(Time.deltaTime);

        Bounds bounds = new Bounds(Vector3.zero, renderBounds);
        particleMaterial.SetBuffer("_Particles", particleBuffer);

        // 【LOGIC FIX】 Always render the buffer that was the DESTINATION of the simulation.
        // After the swap, its index is now stored in pingPongA.
        particleMaterial.SetBuffer("_AliveIndices", aliveIndicesBuffers[pingPongA - 1]);
        
        Graphics.DrawMeshInstancedIndirect(particleMesh, 0, particleMaterial, bounds, indirectArgsBuffer);

        if (Time.frameCount % 60 == 0)
        {
            uint[] finalCounters = new uint[3];
            countersBuffer.GetData(finalCounters);
            // After the swap, the final count is in the counter indexed by pingPongA
            Debug.Log($"🔥 GPU粒子系统运行中 - 活粒子: {finalCounters[pingPongA]}, 死粒子: {finalCounters[0]}, 本帧发射: {Mathf.RoundToInt(emissionRate * Time.deltaTime)}");
        }
    }
    
    // RunSimulationStep remains the same as the last version
    private void RunSimulationStep(float deltaTime)
    {
        computeShader.SetBuffer(kernelUpdateArgs, "_Counters", countersBuffer);
        computeShader.SetBuffer(kernelUpdateArgs, "_IndirectArgs", indirectArgsBuffer);
        computeShader.SetBuffer(kernelEmit, "_Particles", particleBuffer);
        computeShader.SetBuffer(kernelEmit, "_DeadPool", deadPoolBuffer);
        computeShader.SetBuffer(kernelEmit, "_Counters", countersBuffer);
        computeShader.SetBuffer(kernelSimulate, "_Particles", particleBuffer);
        computeShader.SetBuffer(kernelSimulate, "_DeadPool", deadPoolBuffer);
        computeShader.SetBuffer(kernelSimulate, "_Counters", countersBuffer);
        computeShader.SetBuffer(kernelSimulate, "_IndirectArgs", indirectArgsBuffer);

        computeShader.SetInt("_PingPong_A", pingPongA);
        computeShader.SetInt("_PingPong_B", pingPongB);
        computeShader.Dispatch(kernelUpdateArgs, 1, 1, 1);
        
        int emissionCount = Mathf.RoundToInt(emissionRate * deltaTime);
        if (emissionCount > 0)
        {
            computeShader.SetBuffer(kernelEmit, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]);
            computeShader.SetInt("_EmissionCount", emissionCount);
            computeShader.SetVector("_Seeds", new Vector3(Random.value, Random.value, Random.value));
            computeShader.SetInt("_PingPong_A", pingPongA);
            int emitThreadGroups = Mathf.CeilToInt((float)emissionCount / THREAD_COUNT_1D);
            computeShader.Dispatch(kernelEmit, emitThreadGroups, 1, 1);
        }

        uint[] currentCounters = new uint[3];
        countersBuffer.GetData(currentCounters);
        uint simulationCount = currentCounters[pingPongA];

        if (simulationCount > 0)
        {
            computeShader.SetBuffer(kernelSimulate, "_AliveIndices_A", aliveIndicesBuffers[pingPongA - 1]);
            computeShader.SetBuffer(kernelSimulate, "_AliveIndices_B", aliveIndicesBuffers[pingPongB - 1]);
            computeShader.SetFloat("_DeltaTime", deltaTime);
            computeShader.SetInt("_PingPong_A", pingPongA);
            computeShader.SetInt("_PingPong_B", pingPongB);
            int simulateThreadGroups = Mathf.CeilToInt((float)simulationCount / THREAD_COUNT_1D);
            computeShader.Dispatch(kernelSimulate, simulateThreadGroups, 1, 1);
        }

        int temp = pingPongA;
        pingPongA = pingPongB;
        pingPongB = temp;
    }
    
    // The rest of the file (SetShaderParameters, InitializeBuffers, etc.) remains unchanged.
    private void SetShaderParameters()
    {
        computeShader.SetFloat("_MinLifetime", minLifetime);
        computeShader.SetFloat("_MaxLifetime", maxLifetime);
        computeShader.SetFloat("_MinInitialSpeed", minInitialSpeed);
        computeShader.SetFloat("_MaxInitialSpeed", maxInitialSpeed);
        computeShader.SetVector("_EmitterPosition", emitterPosition);
        computeShader.SetVector("_EmitterDirection", emitterDirection.normalized);
        computeShader.SetFloat("_EmitterRadius", emitterRadius);
        computeShader.SetInt("_AffectedByGravity", affectedByGravity ? 1 : 0);
        computeShader.SetInt("_EmissionType", (int)emissionType);
        computeShader.SetFloat("_SpiralForce", spiralForce);
        computeShader.SetFloat("_SpiralRadius", spiralRadius);
        computeShader.SetFloat("_SpiralHeight", spiralHeight);
        computeShader.SetFloat("_SpiralTurns", spiralTurns);
    }
    
    void InitializeBuffers()
    {
        particleBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(float) * 16);
        deadPoolBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint));
        aliveIndicesBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint));
        aliveIndicesBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, ParticleConstants.MAX_PARTICLES, sizeof(uint));
        aliveIndicesBuffers = new GraphicsBuffer[] { aliveIndicesBufferA, aliveIndicesBufferB };
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
        uint[] counters = new uint[] { (uint)ParticleConstants.MAX_PARTICLES, 0, 0 };
        countersBuffer.SetData(counters);
        uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
        if (particleMesh != null)
        {
            args[0] = particleMesh.GetIndexCount(0);
        }
        indirectArgsBuffer.SetData(args);
    }
    
    void SetupCamera()
    {
        if (Camera.main != null)
        {
            Camera.main.backgroundColor = Color.black;
            Camera.main.clearFlags = CameraClearFlags.SolidColor;
            Camera.main.transform.position = new Vector3(0, 6, -15);
            Camera.main.transform.rotation = Quaternion.Euler(15, 0, 0);
        }
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