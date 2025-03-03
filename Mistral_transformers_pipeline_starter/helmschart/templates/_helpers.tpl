{{- define "my-helm-chart.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "my-helm-chart.selectorLabels" -}}
{{- include "my-helm-chart.labels" . -}}
{{- end -}}

{{- define "my-helm-chart.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "my-helm-chart.selectorLabels" . }}
{{- end -}}