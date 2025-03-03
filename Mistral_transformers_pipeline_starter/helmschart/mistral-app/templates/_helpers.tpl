apiVersion: v1
kind: Service
metadata:
  name: {{ include "mistral-app.fullname" . }}
  labels:
    {{- include "mistral-app.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
  selector:
    {{- include "mistral-app.selectorLabels" . | nindent 4 }}