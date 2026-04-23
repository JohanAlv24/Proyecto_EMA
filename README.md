# Clasificación de Series de Tiempo con Representaciones Latentes (VAE)

## Descripción

Este proyecto tiene como objetivo comparar el desempeño en la clasificación de series de tiempo univariadas mediante su proyección a un espacio latente utilizando Codificadores Automáticos Variacionales (VAE), frente a métodos de referencia tradicionales.

Se consideran dos enfoques base:
Un método clásico ampliamente utilizado en la literatura: DTW + KNN
Un enfoque de línea base inferior: clasificación directa sobre las series sin transformación

El proyecto tiene un enfoque académico, utilizando datasets del reposorio UCR.

---

## Motivación

El uso de representaciones latentes busca capturar estructuras más abstractas en las series de tiempo, potencialmente mejorando la separabilidad entre clases. Sin embargo, este enfoque puede resultar limitado frente a métodos más recientes, por lo que este trabajo se enfoca en evaluar su utilidad práctica en un contexto controlado.

---

## Métodos Implementados

DTW (Dynamic Time Warping)
VAE (Variational Autoencoder)
KNN (K-Nearest Neighbors)
SVM (Support Vector Machine)
Decision Trees

---

1. Clona este repositorio:

git clone https://github.com/JohanAlv24/Proyecto_EMA

2. Instala las dependencia:

pip install -r requirements.txt

3. Ejecutar:

python main.py
