### i don't take responsibility if you break your drivers or gpu by building this in its current state



As it currently stands, this is not even barebones, just on the testing phase



The point of this project is to use Vulkan as a backend for Cycles in Blender post 2.93, since that's when they stopped supporting OpenCL, adding support for a bunch of graphics cards in the process



That said, this is mostly a "for me" project, i wanted to learn how things worked so, this is not intended to be used professionally in any way



Yes it would likely be easier to bring support for the modern Blender nodes and features back to 2.93 or forward port that version of Cycles so it works in older graphics cards
But that wouldn't really be studying how code is written for SPIR, Slang, Vulkan, etc. so i'm taking the more complicated approach

a LOT of this is vibe coded or ai-assisted, the approach is to use it as i go to learn and decompose everything so i know what it does until i can write it and debug it myself.



Also im not including my own binaries for the compiled code



Currently, this is not safe to build and run on your machine, as i'm still figuring out raw memory addresses on vulkan myself, spirv-val doesn't see in runtime so if you mess up any arguments eeeh you're on your own







\## The approach is to translate the bare minimum i need from Cycles C++ to Slang and compile to SPIR-V which Vulkan can read, of course not the entire code (why would i translate CUDA and HIP) and have that as it's own standalone thing trying not to break the rest of Blender

