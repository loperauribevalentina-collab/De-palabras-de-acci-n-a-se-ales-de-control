# De-palabras-de-acci-n-a-se-ales-de-control
El lenguaje relacionado con interfaz cerebro-computadora, como cambian los estímulos neuronales en miembros superiores e inferiores al imaginar, leer o visualizar las acciones.

El protocolo experimental integró sincronización de estímulos y adquisición EEG mediante Python (PsychoPy) y el sistema de streaming Lab Streaming Layer (LSL). La interacción entre estímulos, marcadores y señal EEG se resume así:
PsychoPy presentó los estímulos correspondientes a cada paradigma y envió los marcadores del evento.
El Unicorn Hybrid Black envió las señales EEG crudas vía LSL.
Un Lab Recorder consolidó en tiempo real EEG + marcadores en archivos XDF.
Cada trial siguió una estructura temporal compuesta por:

Periodo de descanso (3 s)


Periodo de preparación (1.5 s, con señal acústica)

Ejecución de la tarea (3 s)

Intervalo intertrial (1.5 s)

