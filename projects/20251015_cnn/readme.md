### Entrenamiento CNN Multiclase

Quería probar en mi GPU RTX 2060, en una PC donde eliminé mi instalación de Arch hace 6 meses, así que decidí documentar el proceso completo:

1. **Instalar CUDA y Conda desde los binarios en la página de releases**

   - Descargué CUDA Toolkit compatible con la RTX 2060
   - Instalé Miniconda para gestionar entornos Python aislados

2. **Crear un entorno e instalar dependencias**

   - Configuré un nuevo entorno con Python 3.9
   - Instalé TensorFlow < 2.10 para compatibilidad con CUDA 11.x
   - Agregué dependencias: NumPy, OpenCV, Matplotlib

3. **Intentar encontrar una forma de optimizar el proceso**

   - Probé cargar menos datos en el dataset
   - Ajusté hiperparámetros: batch size, learning rate, epochs

4. **No funcionó, intenté otro enfoque**

   - Problemas de memoria al cargar todo el dataset en RAM
   - Bottleneck en transferencia de datos CPU→GPU
   - Decidí implementar carga dinámica de batches

5. **Usar un script nuevo optimizado para cargar lotes desde CPU a los 6GB de RAM de la GPU**
   - Implementé generador de datos con `tf.data.Dataset`
   - Optimicé pipeline: prefetch, cache parcial, batch paralelo
   - Reducción de MUCHO tiempo de entrenamiento

### Primera corrida exitosa con la GPU

Leí sobre el problema de la pata de hormiga en un paper de investigacion, use como minimo 128x128.

Tres bloques convulcionales de profundidad progresiva 32->64->128

32 de batch
20 epocas

Resultados:
![alt text](r1.png)

Despues de probarlo, marcaba cosas raras como mariquitas como tortugas, tortugas como perros, pero como fue la primera despues de mucho intentar con la grafica no me decepciono, probando con imagenes aleatorias que si son parte del dataset no hubo ningun error.

![alt text](t1.png)

### Segunda

64 batch
50 ep
