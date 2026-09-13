import { mount } from '@vue/test-utils'
import TrendChart from './TrendChart.vue'

vi.mock('vue-echarts', () => ({
  default: {
    name: 'VChart',
    props: ['option'],
    template: '<div class="v-chart-stub" />',
  },
}))

describe('TrendChart', () => {
  it('connects observed windows while retaining their game counts', () => {
    const wrapper = mount(TrendChart, {
      props: {
        series: [{
          deck_label: 'synthetic_deck',
          deck_hash: 'synthetic_hash',
          display_name: 'Synthetic deck',
          points: [
            {
              minute_index: 1,
              ended_at_utc: '2026-01-01T00:01:00Z',
              games: 3,
              score_rate: 0.5,
            },
            {
              minute_index: 2,
              ended_at_utc: '2026-01-01T00:02:00Z',
              games: 0,
              score_rate: null,
            },
            {
              minute_index: 3,
              ended_at_utc: '2026-01-01T00:03:00Z',
              games: 5,
              score_rate: 0.6,
            },
          ],
        }],
      },
    })

    const chart = wrapper.findComponent({ name: 'VChart' })
    const option = chart.props('option') as {
      series: Array<{ data: Array<[string, number, number, number]> }>
    }

    expect(option.series[0].data).toEqual([
      ['2026-01-01T00:01:00Z', 0.5, 3, 0.5],
      ['2026-01-01T00:03:00Z', 0.6, 5, 0.6],
    ])
  })
})
