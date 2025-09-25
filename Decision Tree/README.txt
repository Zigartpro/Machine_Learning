Datos Obtenidos.
Evaluación del último modelo:
Exactitud: 0.961
Z Score: -0.365

Matriz de confusión:
[[388  23]
 [  8 381]]

Reporte de clasificación:
              precision    recall  f1-score   support

         Ham       0.98      0.94      0.96       411
        Spam       0.94      0.98      0.96       389

    accuracy                           0.96       800
   macro avg       0.96      0.96      0.96       800
weighted avg       0.96      0.96      0.96       800

PRUEBA
Con datos completamente nuevos:
Exactitud: 0.967
Matriz de confusión:
[[61  3]
 [ 2 84]]

10 Predicciones individuales del árbol:
--------------------------------------------------
Correo 1: Real=Spam | Predicción=Spam | ✓
Correo 2: Real=Spam | Predicción=Spam | ✓
Correo 3: Real=Ham  | Predicción=Ham  | ✓
Correo 4: Real=Spam | Predicción=Spam | ✓
Correo 5: Real=Ham  | Predicción=Ham  | ✓
Correo 6: Real=Spam | Predicción=Ham  | ✗
Correo 7: Real=Spam | Predicción=Spam | ✓
Correo 8: Real=Spam | Predicción=Spam | ✓
Correo 9: Real=Spam | Predicción=Spam | ✓
Correo 10: Real=Ham  | Predicción=Spam | ✗

Total datos nuevos probados: 150

Conclusión.
Modelo robusto: Mantiene rendimiento con datos nuevos
Sin sobreajuste: Funciona igual o mejor con datos no vistos
Errores balanceados: No tiene sesgo hacia HAM o SPAM
Confiabilidad: Solo 5 errores en 150 correos nuevos
