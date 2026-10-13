As it currently stands, this is not even barebones, just on the testing phase

The point of this project is to use Vulkan as a backend for Cycles in Blender post 2.93, since that's when they stopped supporting OpenCL, adding support for a bunch of graphics cards in the process

That said, this is mostly a "for me" project, i wanted to learn how things worked so, this is not intended to be used professionally in any way



Yes it would likely be easier to bring support for the modern Blender nodes and features back to 2.93 or forward port that version of Cycles so it works in older graphics cards
But that wouldn't really be studying how code is written for SPIR, Slang, Vulkan, etc. so i'm taking the more complicated approach

a LOT of this is vibe coded or ai-assisted, the approach is to use it as i go to learn and decompose everything so i know what it does until i can write it and debug it myself.

for this v0 there's only a very simple .slang script that's compiled in (you wouldn't guess it) Slang

copy this in your bash terminal:



slangc chase.slang -target spirv -entry main -stage compute -o chase.spv \[actual compilation]



spirv-val --target-env vulkan1.3 chase.spv \[spir validation for vulkan 1.3]



spirv-dis chase.spv > chase.txt \[dissasembly binary to a readable assembly text file]



Also im not including my own binaries for the compiled chase.spv as that could have some personal information about my pc, compile it for your own

Same with chase.txt





\#################################################################################################

HOLY SHIT CHECK YOUR ARGUMENTS BEFOREHAND AND NEVER RUN RAW ADRESSES THAT CAN CORRUPT THE DRIVER



DO NOT, I REPEAT, DO NOT WRITE BEYOND THE INTENDED BUFFER, RAW GPU POINTERS HAVE NO BOUNDS CHECKING



SPIRV-VAL CANNOT SEE WITHIN RAW MEMORY ADDRESSES ON RUNTIME, IT CAN AND WILL BREAK GPU MEMORY ADDRESSES

\#################################################################################################



Currently, this is not safe to build and run on your machine, as i'm still figuring out memory addresses myself, i don't take responsibility if you break your drivers...

...or gpu by building this in its current state

