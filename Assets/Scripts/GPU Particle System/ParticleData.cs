using UnityEngine;

[System.Serializable]
public struct Particle
{
    public Vector4 lifetime;  // x: current, y: max
    public Vector4 velocity;  // xyz: velocity, w: unused
    public Vector4 position;  // xyz: position, w: unused
    public Vector4 color;     // rgba: color
}

public static class ParticleConstants
{
    public const int MAX_PARTICLES = 1000000;  // 原版：100万粒子
    public const int LOCAL_SIZE = 32;          // 原版：32
    public const int GRADIENT_SAMPLES = 32;    // 原版：32
    public const float CAMERA_FAR_PLANE = 1000.0f; // 原版：1000
}