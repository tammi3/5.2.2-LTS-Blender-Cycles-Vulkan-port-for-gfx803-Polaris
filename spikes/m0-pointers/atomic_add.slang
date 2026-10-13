struct Params
{
    Atomic<uint>* target;  // the float's bits, stored as a uint
    float value;           // what each thread wants to add
};

[[vk::push_constant]] ConstantBuffer<Params> params;

// Emulated float atomic add using compare-and-swap (CAS).
float atomic_add_float(Atomic<uint>* addr, float v)
{
    uint expected = addr->load();
    while (true)
    {
        // Compute the new value from what we last saw...
        uint desired = asuint(asfloat(expected) + v);
        // ...and write it only if nobody changed it meanwhile.
        uint original = addr->compareExchange(expected, desired);
        if (original == expected)
            return asfloat(original);  // we won the race
        expected = original;           // somebody else changed it; retry, this makes it very slow
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main()
{
    atomic_add_float(params.target, params.value);
}

// emulating float atomic add is a lot slower, for future updates or doing it for more graphics card...
// ...consider checking if the graphics card has shaderBufferFloat32AtomicAdd and use that instead