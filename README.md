# perceptron
Implementación de un perceltron multicapa haciendo uso de la librería armadillo.

1. Se realizará un entrenamiento en la red para aprender las vocales (A, E, I, O, U) en mayúscula.
  1.1 Cada vocal es representada como un vector binario.
  1.2 Cada vocal tendrá asignado un valor que la representará en la matrix objetivo. Asignados como:
    A --> 0.0
    E --> 0.2
    I --> 0.4
    O --> 0.6
    U --> 0.8
2. Se realizará una simulación utilizando una entrada ruidosa seleccionando una vocal deformada.
  2.1 Se eliminan valores de 1 de la vocal para simular ruido.
  2.2 Se probará la vocal ruidosa con los nuevos pesos obtenidos en la estapa de entrenamiento.
3. La red tiene la siguiente configuración:
  - Error para convergencia de 0.00001.
  - Tasa de aprendizaje de 0.4.
  - Función de activación signoidal y derivada.
  - Tres neuronas en la capa oculta.
  
Notas
Contenido:
 - Código (/src)
 - Entrada (entrada.txt)
 - Salida (resultado.txt)
