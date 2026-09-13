export const percent = (value: number | null | undefined, digits = 1) =>
  value == null ? '—' : `${(value * 100).toFixed(digits)}%`

export const count = (value: number | null | undefined) =>
  value == null ? '—' : value.toLocaleString('en-US')

export const decimal = (value: number | null | undefined, digits = 4) =>
  value == null ? '—' : value.toFixed(digits)

export const gibibytes = (value: number | null | undefined) =>
  value == null ? '—' : `${(value / 1024 ** 3).toFixed(1)} GiB`

export const duration = (value: number | null | undefined) =>
  value == null ? '—' : value < 60 ? `${value.toFixed(1)}s` : `${(value / 60).toFixed(1)}m`

export const relativeAge = (value: number | null | undefined) => {
  if (value == null) return '—'
  if (value < 60) return `${Math.round(value)}s`
  if (value < 3600) return `${Math.round(value / 60)}m`
  return `${(value / 3600).toFixed(1)}h`
}

export const compactHash = (value: string | null | undefined, length = 12) =>
  value ? value.slice(0, length) : '—'

export const dateTime = (value: string | null | undefined) =>
  value
    ? new Intl.DateTimeFormat('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      }).format(new Date(value))
    : '—'
