## Objetivos

+ Simple Energy RGB
- Rotar la imagen en vez de usar horizontal seams  -> es una tontería
- Distintos tipos de de energía
	+ eHOG
	- Forward Energy
+ Optimización con escalado previo
- Optimizar el código en general -> creo que hacerlo con una matriz está bien y no hace falta usar punteros
- Protección de objetos con máscaras de bits
- Eliminación de objetos con máscaras de bits
- Probar con muchas imágenes
- MEMORIA!!!!!


## Enlaces útiles

Paper donde explican el paper original
http://www.aui.ma/sse-capstone-repository/pdf/spring2016/Content%20Aware%20Image%20Resizing%20Using.pdf

Otra explicación y el código correspondiente en python
(útil para forward energy y object removal)
https://andrewdcampbell.github.io/seam-carving
https://github.com/andrewdcampbell/seam-carving/blob/master/seam_carving.py

Implementación en python (incluye forward energy)
https://github.com/vivianhylee/seam-carving/blob/01404fdce6b7613fa3a64eb1227fbfa1b533f52e/seam_carving.py

Implementación en python
https://github.com/jordan-bonecutter/SeamCarver/blob/master/seam_carver.py

Explicación sin código
http://cs.brown.edu/courses/cs129/results/proj3/taox/

Explicación + implementación básica
https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
