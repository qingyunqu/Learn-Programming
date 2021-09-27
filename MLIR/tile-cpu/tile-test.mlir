#strided1D = affine_map<(d0) -> (d0)>
#strided2D = affine_map<(d0, d1)[] -> (d0, d1)>

// Creates and returns a 1-D buffer of size %s filled with the value %f
func @alloc_filled_f32(%s : index, %f : f32) -> memref<?xi8> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %s4 = muli %s, %c4: index
  %buf = memref.alloc(%s4) {alignment = 256} : memref<?xi8>
  %V = memref.view %buf[%c0][%s] : memref<?xi8> to memref<?xf32, #strided1D>
  linalg.fill(%f, %V) : f32, memref<?xf32, #strided1D>
  return %buf : memref<?xi8>
}

func @matmul() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index

  %m = constant 1024 : index
  %k = constant 1024 : index
  %n = constant 1024 : index
  %mk = constant 1048576 : index
  %kn = constant 1048576 : index
  %mn = constant 1048576 : index

  %f1 = constant 1.00000e+00 : f32
  %f2 = constant 2.00000e+00 : f32
  %f10 = constant 10.00000e+00 : f32

  %bA = call @alloc_filled_f32(%mk, %f2) : (index, f32) -> (memref<?xi8>)
  %bB = call @alloc_filled_f32(%kn, %f1) : (index, f32) -> (memref<?xi8>)
  %bC = call @alloc_filled_f32(%mn, %f10) : (index, f32) -> (memref<?xi8>)

  %A = memref.view %bA[%c0][%m, %k] : memref<?xi8> to memref<?x?xf32, #strided2D>
  %B = memref.view %bB[%c0][%k, %n] : memref<?xi8> to memref<?x?xf32, #strided2D>
  %C = memref.view %bC[%c0][%m, %n] : memref<?xi8> to memref<?x?xf32, #strided2D>

  linalg.matmul
    ins(%A, %B: memref<?x?xf32, #strided2D>, memref<?x?xf32, #strided2D>)
	outs(%C: memref<?x?xf32, #strided2D>)
  %res = memref.load %C[%c6, %c7] : memref<?x?xf32, #strided2D>

  memref.dealloc %bC : memref<?xi8>
  memref.dealloc %bB : memref<?xi8>
  memref.dealloc %bA : memref<?xi8>

  return %res : f32
}
