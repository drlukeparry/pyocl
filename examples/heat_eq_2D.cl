//#define NON_OPTIMIZED 1
//#define
// Workshop - https://cvw.cac.cornell.edu/gpu/coalesced
// Numerical techniques - http://folk.ntnu.no/leifh/teaching/tkt4140/._main057.html
kernel void copy(global const float * in,
                 global float * out)
{
    int nx = get_global_size(0);
    int ny = get_global_size(1);
    
    int x = get_global_id(0); // (x,y) = pixel to process
    int y = get_global_id(1); // in this work item
    
    float a = in[x+y*nx];           // Load
    
    float b = 3.9f * a * (1.0f-a); // Three floating point ops
    
    out[x+y*nx] = b; // Load and Store 
}


__kernel void heat_eq_2D_shared4(__global float *u1, __global const float *u0, __local float *block, float alpha, float dt, float dx, float dy) {
    
    //Get total number of cells
    int nx = get_global_size(0);
    int ny = get_global_size(1);
    
    int i  = get_global_id(0);
    int j  = get_global_id(1);
    
    int loadWidth = 8;
    
    int localHeight = 18; //18; 
    int localWidth  = 24;// vec4 needs to be a multiple of 4
    
    // Determine where each workgroup begins reading
    int groupStartCol = get_group_id(0)*get_local_size(0)/loadWidth;
    int groupStartRow = get_group_id(1)*get_local_size(1);
    
    // Determine the local ID of each work-item
    int localId = get_local_id(1) * get_local_size(0) + get_local_id(0); // Flatten id to array
    
    // Each work item is reading 
    int localRow = localId / (localWidth / loadWidth); // Multiple of 4 to get the current row
    int localCol = localId % (localWidth /loadWidth);  // Position along using the modulus
    
    // Current position in the global grid
    int globalRow = groupStartRow + localRow;
    int globalCol = groupStartCol + localCol;
 
    
    // Determine the global ID of each work-item. work-items
    // representing the output region will have a unique
    // global ID
    
    
    //float block[18*18];
    
    // Cache global memory into local memory
    // We set the block4
    
     __local float8* block4;

     
     // Initialise the block 4 pointer to the local memory space
     block4 = (__local float8*) &block[localRow*localWidth+localCol*loadWidth];
    
     if(globalRow < ny && globalCol < nx/loadWidth && 
         localRow < localHeight) {
         block4[0] = vload8(globalRow*nx/loadWidth +globalCol, u0);
     }
    
    __local float kappa1;
    __local float kappa2;
    
    if(get_local_id(0) == 0 && get_local_id(1) == 0) {
        kappa1 = alpha * dt / (dx * dx);
        kappa2 = alpha * dt / (dy * dy);
        
    }
    
    // Required to ensure all the work items are syncrhonised together
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //i = clamp(i,1,nx-2);
    //j = clamp(j,1,ny-2);
    //Calculate the four indices of our neighbouring cells
    //int offset = 1; 
    
    // The initial offset is needed so Image[i,j] starts at (1,1)
    
    localCol = get_local_id(0);
    localRow = get_local_id(1);
    
    int ii = localCol+1; int jj = localRow+1;
    
    int center = jj * localWidth + (ii);
    int north = (jj + 1) * localWidth + (ii);
    int south = (jj - 1) * localWidth + ii;
    int east = jj * localWidth + (ii + 1);
    int west = jj * localWidth + (ii - 1);
    
    float tmp = 0.0;
    

    
    globalCol = get_group_id(0)*get_local_size(0) + localCol;
    globalRow =  get_group_id(1)*get_local_size(1) + localRow;
    
    //Internall cells
    if ((globalCol < nx-2) && (globalRow < ny-2)) {
        u1[(j+1)* nx + i+1] = block[center] + kappa1 * (block[west] - 2.0 * block[center] + block[east])
        + kappa2 * (block[south] - 2.0 * block[center] + block[north]);
        // u1[(j)* nx + i]= tmp;
    } 
    
    if( globalCol < 1 || globalCol > nx-2 || globalRow < 1 || globalRow > ny-2) {
        u1[(j)* nx + i] = block[localRow*localWidth + localCol];
    }
    
    
}

__kernel void heat_eq_2D_shared(__global float *u1, __global const float *u0, __local float *block, float alpha, float dt, float dx, float dy) {

   //Get total number of cells
   int nx = get_global_size(0);
   int ny = get_global_size(1);

   int i  = get_global_id(0);
   int j  = get_global_id(1);

    
    // Determine where each workgroup begins reading
    int groupStartCol = get_group_id(0)*get_local_size(0);
    int groupStartRow = get_group_id(1)*get_local_size(1);

    // Determine the local ID of each work-item
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);

    int localHeight = 18; //18; 
    int localWidth  = 18;// vec4 needs to be a multiple of 4

    // Determine the global ID of each work-item. work-items
    // representing the output region will have a unique
    // global ID

    int globalCol = groupStartCol + localCol;
    int globalRow = groupStartRow + localRow;
   
    //float block[18*18];
    
    // Cache global memory into local memory
    // The iterator splits the local grid into blocks matching the chosen workgroup size
    for(int jj = localRow; jj < localHeight; jj += get_local_size(1)) {
        // Current row is iterated by local workgroup size (y dir)
        int curRow = groupStartRow  + jj;

        // Iterate across the y direction
        for(int ii = localCol; ii < localWidth; ii += get_local_size(0)) {
            int curCol = groupStartCol + ii;

            // Check if the current workitem is in bounds of the computational domain
            if(curCol < nx && curRow < ny ) {
                // Copy the globalb meory to the local shared memory
                block[jj * localWidth + ii] = u0[curRow * nx + curCol];
            }
        }
    }
   

    
   __local float kappa1;
   __local float kappa2;

   if(get_local_id(0) == 0 && get_local_id(1) == 0) {
       kappa1 = alpha * dt / (dx * dx);
       kappa2 = alpha * dt / (dy * dy);

   }
   
   // Required to ensure all the work items are syncrhonised together
   barrier(CLK_LOCAL_MEM_FENCE);

    //i = clamp(i,1,nx-2);
   //j = clamp(j,1,ny-2);
   //Calculate the four indices of our neighbouring cells
   //int offset = 1; 
   int ii = localCol+1; int jj = localRow+1;
   
   int center = jj * localWidth + (ii);
   int north = (jj + 1) * localWidth + (ii);
   int south = (jj - 1) * localWidth + ii;
   int east = jj * localWidth + (ii + 1);
   int west = jj * localWidth + (ii - 1);
   
    float tmp = 0.0;

    
   //Internall cells
   if ((globalCol < nx-2) && (globalRow < ny-2)) {
       u1[(j+1)* nx + i+1] = block[center] + kappa1 * (block[west] - 2.0 * block[center] + block[east])
                                    + kappa2 * (block[south] - 2.0 * block[center] + block[north]);
      // u1[(j)* nx + i]= tmp;
   } 
   
   if( globalCol < 1 || globalCol > nx-2 || globalRow < 1 || globalRow > ny-2) {
       u1[(j)* nx + i] = block[localRow*localWidth + localCol];
   }
   

}


// vec4 needs to be a multip
__kernel void heat_eq_2D(__global float *u1, __global const float *u0, float alpha, float dt, float dx, float dy) {
    
    //Get total number of cells
    int nx = get_global_size(0);
    int ny = get_global_size(1);
    
    int i  = get_global_id(0);
    int j  = get_global_id(1);
    
    __local float kappa1;
    __local float kappa2;
    
    if(get_local_id(0) == 0 && get_local_id(1) == 0) {
        kappa1 = alpha * dt / (dx * dx);
        kappa2 = alpha * dt / (dy * dy);
        
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //i = clamp(i,1,nx-2);
    //j = clamp(j,1,ny-2);
    //Calculate the four indices of our neighbouring cells
    int center = j * nx + i;
    int north = (j + 1) * nx + i;
    int south = (j - 1) * nx + i;
    int east = j * nx + (i + 1);
    int west = j * nx + (i - 1);
    
    
    //Internall cells
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        u1[center] = u0[center] + kappa1 * (u0[west] - 2.0 * u0[center] + u0[east])
        + kappa2 * (u0[south] - 2.0 * u0[center] + u0[north]);
    } else
    {
        // Boundary conditions (ghost cells)\n
        u1[center] = u0[center];
    }
    
    
}
