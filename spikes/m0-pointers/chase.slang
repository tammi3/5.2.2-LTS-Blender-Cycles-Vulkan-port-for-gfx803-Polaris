
struct Node
{
    uint value;
    Node* next;   // a real pointer, not an index
};

struct Params
{
    Node* head;    // address of the first node
    uint* output;  // where to write the result
};

[[vk::push_constant]] ConstantBuffer<Params> params; // fast push constant memory

[shader("compute")]
[numthreads(1, 1, 1)]

void main()
{
    uint sum = 0;
    Node* n = params.head;
    // Walk the list until we hit a null pointer.
    while (n != nullptr)
    {
        sum += n->value;
        n = n->next;
    }
    *params.output = sum;
}