# Implementación de un Perceptrón para Clasificación de Solicitudes de Préstamo

## Descripción
Este proyecto implementa y entrena un perceptrón en Python para clasificar solicitudes de préstamo en aprobadas (1) o rechazadas (0). Se centra en reforzar el conocimiento en aprendizaje supervisado, ajuste de pesos en redes neuronales simples y la implementación de modelos de clasificación binaria.

## Problema a Resolver
Una institución financiera desea automatizar la clasificación de solicitudes de préstamo, utilizando un perceptrón que evalúe cuatro factores clave para tomar decisiones:
- **Puntaje de crédito**: Valor entre 300 y 850.
- **Ingresos mensuales**: Expresado en miles de pesos.
- **Monto del préstamo solicitado**: Expresado en miles de pesos.
- **Relación deuda/ingresos**: Valor decimal (por ejemplo, 0.2, 0.5, etc.).

La institución proporciona un conjunto de datos históricos con ejemplos de solicitudes aprobadas y rechazadas. El perceptrón debe aprender a clasificar correctamente cada solicitud.

## Objetivo
Entrenar al perceptrón para que, a partir de estos valores, clasifique correctamente si la solicitud debe ser aprobada o rechazada.

## Código
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Datos históricos (Ejemplo de tabla adjunta)
datos = np.array([
    [750, 50, 200, 0.3, 1],  # Aprobado
    [600, 30, 150, 0.5, 0],  # Rechazado
    [720, 40, 180, 0.4, 1],  # Aprobado
    [580, 25, 120, 0.6, 0],  # Rechazado
    [680, 35, 170, 0.45, 1], # Aprobado
    [550, 20, 100, 0.7, 0]   # Rechazado
])

# Separar características y etiquetas
X = datos[:, :-1]
y = datos[:, -1]

# Normalizar datos para mejorar el aprendizaje
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Inicialización de parámetros
learning_rate = 0.1
epochs = 20
weights = np.random.rand(X.shape[1])
bias = np.random.rand()

def activation_function(x):
    return 1 if x >= 0 else 0

# Entrenamiento del perceptrón
for epoch in range(epochs):
    print(f"Época: {epoch + 1}")
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = activation_function(linear_output)
        error = y[i] - prediction
        
        # Actualización de pesos y bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error
        
        print(f"Muestra {i+1}: Entrada {X[i]}, Esperado {y[i]}, Predicción {prediction}, Error {error}")
    print(f"Pesos: {weights}, Bias: {bias}\n")

# Prueba con nuevos datos
nuevos_datos = np.array([[700, 45, 190, 0.35]])
nuevos_datos = scaler.transform(nuevos_datos)
resultado = activation_function(np.dot(nuevos_datos, weights) + bias)
print("Solicitud aprobada" if resultado == 1 else "Solicitud rechazada")
```
## Estructura del Proyecto
```
📂 Perceptron-Automatizacion-Solicitudes
│── 📂 data  # Datos generados para las pruebas
│── 📂 results  # Resultados y análisis
│── 📄 README.md  # Documentación del proyecto
│── 📄 Implementacion_Perceptron.ipynb  # Implementación en Jupyter Notebook
```

## Requisitos
Para ejecutar el proyecto, necesitas tener instalado:
- **Python 3.x**
- **Jupyter Notebook**
- **Librerías necesarias** (se pueden instalar con pip):
  ```bash
  pip install notebook
  ```
## Bibliotecas necesarias:
 ```bash
numpy
pandas
scikit-learn
```
Para instalar las dependencias, ejecute: ```bash pip install numpy pandas scikit-learn```

## Uso
1. Clona este repositorio:
   ```bash
   git clone https://github.com/Jair-Artreaga/IPerceptron-Automatizacion-Solicitudes.git
   ```
2. Accede al directorio del proyecto:
   ```bash
   cd Implementacion_Perceptron
   ```
3. Ejecuta Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Abre el archivo `Implementacion_Perceptron.ipynb` y ejecuta las celdas para ver los resultados.

## Contribución
Si desea mejorar el código o agregar nuevas funcionalidades, puede hacer un fork del repositorio y enviar un pull request.

## Autor
**[Roberto Jair Arteaga Valenzuela]**
