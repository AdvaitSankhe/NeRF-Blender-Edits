# Editable NeRF Blender

NeRFs reconstruct scenes well, but they’re hard to edit. This project is an attempt to make NeRF scenes more transparent and editable.

## What we did

- Extracted an explicit mesh from a trained NeRF using marching cubes  
- Used Segment Anything (SAM) on selected views to isolate objects  
- Mapped 2D masks back into 3D to separate individual objects  
- Refined the isolation with a small optimization step  
- Edited objects in Blender and retrained NeRF on the modified scene  

Tested on the synthetic drum scene (e.g., flipping the drum seat, removing the bass drum).

The focus is geometry-level editing and object-level control inside a NeRF scene.
