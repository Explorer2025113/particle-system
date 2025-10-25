using UnityEngine;
using System.Runtime.InteropServices; // 需要添加这个 using

[System.Serializable]
[StructLayout(LayoutKind.Sequential)] // 添加这个属性以确保内存布局一致
public struct Particle
{
    public Vector4 lifetime;  // x: current, y: max
    public Vector4 velocity;  // xyz: velocity
    public Vector4 position;  // xyz: position
    public Vector4 color;     // rgba: color
    public Vector4 attributes; // x: emitterId, yzw: unused
    public uint globalId;     // 全局唯一ID
    public uint padding1;    // 内存对齐
    public uint padding2;    // 内存对齐
    public uint padding3;     // 内存对齐
}

public static class ParticleConstants
{
    public const int MAX_PARTICLES = 1000000;  // 原版：100万粒子
    public const int LOCAL_SIZE = 32;          // 原版：32
    public const int GRADIENT_SAMPLES = 32;    // 原版：32
    public const float CAMERA_FAR_PLANE = 1000.0f; // 原版：1000
}