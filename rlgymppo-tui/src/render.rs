use std::collections::{HashMap, VecDeque};

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

pub(crate) type MetricHistory = HashMap<String, VecDeque<f64>>;

#[derive(Default)]
pub(crate) struct LayoutPlanCache {
    cached: Option<CachedLayoutPlan>,
}

#[derive(Clone)]
struct CachedLayoutPlan {
    key: LayoutCacheKey,
    columns: Vec<CachedColumnPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LayoutCacheKey {
    width: u16,
    height: u16,
    groups: Vec<GroupLayoutKey>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GroupLayoutKey {
    name: String,
    key_prefix: String,
    entry_names: Vec<String>,
    width: u16,
    height: u16,
}

#[derive(Clone)]
struct CachedColumnPlan {
    group_indices: Vec<usize>,
    width: u16,
    height: u16,
}

pub(crate) const SPARKLINE_HISTORY_LEN: usize = 12;
const SPARKLINE_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

#[cfg(test)]
thread_local! {
    static SEARCH_NODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn search_nodes() -> usize {
    SEARCH_NODES.with(|counter| counter.get())
}

#[cfg(test)]
fn reset_search_nodes() {
    SEARCH_NODES.with(|counter| counter.set(0));
}

use crate::format::{display_width, format_num, metric_value_display, metric_value_for_sparkline};

/// Category grouping information for the display layout.
#[derive(Clone)]
pub(crate) struct MetricGroup {
    pub(crate) name: String,
    pub(crate) key_prefix: String,
    pub(crate) color: Color,
}

/// Pre-defined groups in display order. Metrics whose keys start with a
/// group's `key_prefix + "/"` are shown under that group.
struct KnownMetricGroup {
    name: &'static str,
    key_prefix: &'static str,
    color: Color,
}

const GROUPS: &[KnownMetricGroup] = &[
    KnownMetricGroup {
        name: "Collect",
        key_prefix: "Collect",
        color: Color::Cyan,
    },
    KnownMetricGroup {
        name: "GAE",
        key_prefix: "GAE",
        color: Color::Green,
    },
    KnownMetricGroup {
        name: "Loss",
        key_prefix: "Loss",
        color: Color::Red,
    },
    KnownMetricGroup {
        name: "Update",
        key_prefix: "Update",
        color: Color::Yellow,
    },
    KnownMetricGroup {
        name: "Timing",
        key_prefix: "Timing",
        color: Color::Magenta,
    },
    KnownMetricGroup {
        name: "Throughput",
        key_prefix: "Throughput",
        color: Color::Blue,
    },
    KnownMetricGroup {
        name: "Cumulative",
        key_prefix: "Cumulative",
        color: Color::White,
    },
    KnownMetricGroup {
        name: "Rating",
        key_prefix: "Rating",
        color: Color::LightYellow,
    },
];

fn metric_groups(metrics: &HashMap<String, f64>) -> Vec<MetricGroup> {
    let mut prefixes: Vec<&str> = metrics
        .keys()
        .filter_map(|key| key.rsplit_once('/').map(|(prefix, _)| prefix))
        .collect();
    prefixes.sort_unstable_by_key(|prefix| group_sort_key(prefix));
    prefixes.dedup();

    prefixes
        .into_iter()
        .map(|prefix| {
            if let Some(group) = GROUPS.iter().find(|group| group.key_prefix == prefix) {
                MetricGroup {
                    name: group.name.to_string(),
                    key_prefix: group.key_prefix.to_string(),
                    color: group.color,
                }
            } else {
                MetricGroup {
                    name: prefix.to_string(),
                    key_prefix: prefix.to_string(),
                    color: custom_group_color(prefix),
                }
            }
        })
        .collect()
}

/// Build a sorted list of `(display_name, value)` pairs for a group.
fn group_entries<'a>(metrics: &'a HashMap<String, f64>, prefix: &str) -> Vec<(&'a str, f64)> {
    let mut entries: Vec<(&str, f64)> = metrics
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(&format!("{prefix}/"))
                .filter(|name| !name.contains('/'))
                .map(|name| (name, *v))
        })
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    entries
}

/// Render the header bar (title + key cumulative numbers).
pub(crate) fn render_header(
    frame: &mut ratatui::Frame,
    area: Rect,
    metrics: &HashMap<String, f64>,
) {
    let steps = metrics
        .get("Cumulative/steps")
        .map(|v| format_num(*v))
        .unwrap_or_default();
    let updates = metrics
        .get("Cumulative/updates")
        .map(|v| format_num(*v))
        .unwrap_or_default();

    let header_text = Text::from(Line::from(vec![
        Span::styled(
            " rlgymppo ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(format!("Steps: {steps}"), Style::default().fg(Color::Cyan)),
        Span::raw(" │ "),
        Span::styled(
            format!("Iteration: {updates}"),
            Style::default().fg(Color::Green),
        ),
    ]));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    frame.render_widget(
        Paragraph::new(header_text)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Render all metric groups in a responsive grid.
///
/// The grid chooses the largest content-fitting column count, then greedily packs
/// groups by their desired height so dense groups do not all land in one column.
pub(crate) fn render_metrics_grid(
    frame: &mut ratatui::Frame,
    area: Rect,
    metrics: &HashMap<String, f64>,
    history: &MetricHistory,
    layout_cache: &mut LayoutPlanCache,
    scroll_offset: u16,
    show_sparklines: bool,
) -> u16 {
    let metric_groups = metric_groups(metrics);
    let groups: Vec<MetricGroupView<'_>> = metric_groups
        .iter()
        .filter_map(|group| {
            let entries = group_entries(metrics, &group.key_prefix);
            if entries.is_empty() {
                None
            } else {
                Some(MetricGroupView { group, entries })
            }
        })
        .collect();

    if area.width == 0 || area.height == 0 {
        return 0;
    }

    if groups.is_empty() {
        render_empty_state(frame, area);
        return 0;
    }

    let empty_history = MetricHistory::new();
    let visible_history = if show_sparklines {
        history
    } else {
        &empty_history
    };
    let columns =
        layout_cache.plan_metric_columns(area.width, area.height, &groups, visible_history);

    let content_height = columns
        .iter()
        .map(|column| column.height)
        .max()
        .unwrap_or(0);
    let max_scroll = content_height.saturating_sub(area.height);
    let scroll_offset = scroll_offset.min(max_scroll);

    let total_width: u16 = columns
        .iter()
        .map(|column| column.width)
        .fold(0_u16, u16::saturating_add);
    let extra_width = area.width.saturating_sub(total_width);
    let base_extra = extra_width / columns.len() as u16;
    let extra_remainder = extra_width % columns.len() as u16;
    let mut x = area.x;

    for (idx, column) in columns.iter().enumerate() {
        let remaining_width = area.x.saturating_add(area.width).saturating_sub(x);
        if remaining_width == 0 {
            break;
        }

        let extra = base_extra + u16::from((idx as u16) < extra_remainder);
        let width = column.width.saturating_add(extra).min(remaining_width);
        let col_area = Rect::new(x, area.y, width, area.height);
        render_column(
            frame,
            col_area,
            visible_history,
            &column.groups,
            scroll_offset,
        );
        x = x.saturating_add(width);
    }

    max_scroll
}

/// Render a column of metric groups using content-aware heights.
fn render_column(
    frame: &mut ratatui::Frame,
    area: Rect,
    history: &MetricHistory,
    groups: &[MetricGroupView<'_>],
    scroll_offset: u16,
) {
    if groups.is_empty() {
        return;
    }

    let mut lines = Vec::new();

    for view in groups {
        lines.extend(render_group_lines(
            area.width,
            view.group,
            &view.entries,
            history,
        ));
    }

    frame.render_widget(
        Paragraph::new(Text::from(lines)).scroll((scroll_offset, 0)),
        area,
    );
}

#[derive(Clone)]
struct MetricGroupView<'a> {
    group: &'a MetricGroup,
    entries: Vec<(&'a str, f64)>,
}

#[derive(Clone)]
struct ColumnPlan<'a> {
    groups: Vec<MetricGroupView<'a>>,
    width: u16,
    height: u16,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct LayoutScore {
    overflow: u32,
    wasted_area: u32,
    height_imbalance: u32,
    unused_width: u32,
    column_penalty: u32,
    max_height: u32,
}

impl LayoutPlanCache {
    fn plan_metric_columns<'a>(
        &mut self,
        width: u16,
        height: u16,
        groups: &[MetricGroupView<'a>],
        history: &MetricHistory,
    ) -> Vec<ColumnPlan<'a>> {
        let group_dims: Vec<GroupDimensions> = groups
            .iter()
            .map(|view| GroupDimensions {
                width: required_group_width(view, history),
                height: desired_group_height(view.entries.len()),
            })
            .collect();
        let key = LayoutCacheKey {
            width,
            height,
            groups: group_layout_key(groups, &group_dims),
        };

        if let Some(cached) = self.cached.as_ref().filter(|cached| cached.key == key) {
            return cached
                .columns
                .iter()
                .map(|column| build_column_plan_from_indices(groups, column))
                .collect();
        }

        let cached_columns = plan_metric_columns_uncached(width, height, groups, &group_dims);
        let columns = cached_columns
            .iter()
            .map(|column| build_column_plan_from_indices(groups, column))
            .collect();
        self.cached = Some(CachedLayoutPlan {
            key,
            columns: cached_columns,
        });
        columns
    }
}

fn plan_metric_columns_uncached(
    width: u16,
    height: u16,
    groups: &[MetricGroupView<'_>],
    group_dims: &[GroupDimensions],
) -> Vec<CachedColumnPlan> {
    let mut best: Option<(LayoutScore, Vec<CachedColumnPlan>)> = None;
    let heights: Vec<u16> = group_dims.iter().map(|dims| dims.height).collect();
    let total_height: u32 = heights.iter().map(|&h| h as u32).sum();

    for num_cols in 1..=groups.len() {
        if minimum_total_column_width(group_dims, num_cols) > width {
            break;
        }

        // Once a perfect layout (zero overflow, zero waste, zero height
        // imbalance) is known, any layout with more columns can only improve on
        // it with an equally perfect packing, i.e. all columns ending at the
        // exact same height.  Skip searches whose heights cannot be partitioned
        // into `num_cols` equal sums, and cap the rest at that height so
        // non-perfect branches are pruned immediately.
        let perfect = best.as_ref().is_some_and(|(score, _)| {
            score.overflow == 0 && score.wasted_area == 0 && score.height_imbalance == 0
        });
        if perfect && !can_partition_heights(&heights, num_cols) {
            continue;
        }
        let height_cap = perfect.then(|| (total_height / num_cols as u32) as u16);
        let seed = best.as_ref().map(|(score, _)| *score);

        if let Some((score, columns)) = best_column_assignment(
            groups, group_dims, num_cols, width, height, height_cap, seed,
        ) && best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, columns));
        }
    }

    best.map(|(_, columns)| columns).unwrap_or_else(|| {
        vec![build_cached_column_plan(
            group_dims,
            (0..groups.len()).collect(),
        )]
    })
}

/// Whether the group heights can be partitioned into `k` groups each summing
/// to `sum(heights) / k` exactly.  A layout whose columns all end at the same
/// height (zero wasted area and zero height imbalance) exists iff this holds.
fn can_partition_heights(heights: &[u16], k: usize) -> bool {
    let total: u32 = heights.iter().map(|&h| h as u32).sum();
    if k == 0 || k > heights.len() || !total.is_multiple_of(k as u32) {
        return false;
    }
    let target = total / k as u32;
    if k == 1 {
        return true;
    }
    if heights.iter().any(|&h| h as u32 > target) {
        return false;
    }

    let n = heights.len();
    if n > 18 {
        // Too many heights for the bitmask DP; the per-search height cap still
        // bounds the search, so treat the partition as feasible.
        return true;
    }

    let mut sums = vec![0_u32; 1 << n];
    for mask in 1_usize..(1_usize << n) {
        let lsb = mask & mask.wrapping_neg();
        sums[mask] = sums[mask ^ lsb] + heights[lsb.trailing_zeros() as usize] as u32;
    }
    // 0 = unknown, 1 = partitionable, 2 = not.
    let mut memo = vec![0_u8; 1 << n];

    fn solve(mask: usize, target: u32, sums: &[u32], memo: &mut [u8]) -> bool {
        if mask == 0 {
            return true;
        }
        if memo[mask] != 0 {
            return memo[mask] == 1;
        }
        // A mask whose total is not a multiple of `target` cannot partition.
        if !sums[mask].is_multiple_of(target) {
            memo[mask] = 2;
            return false;
        }
        // The group containing the lowest set bit must sum to `target`.
        let first = mask & mask.wrapping_neg();
        let rest = mask ^ first;
        let mut sub = rest;
        loop {
            let group = sub | first;
            if sums[group] == target && solve(mask ^ group, target, sums, memo) {
                memo[mask] = 1;
                return true;
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & rest;
        }
        memo[mask] = 2;
        false
    }

    solve((1 << n) - 1, target, &sums, &mut memo)
}

fn minimum_total_column_width(group_dims: &[GroupDimensions], num_cols: usize) -> u16 {
    let mut widths: Vec<u16> = group_dims.iter().map(|dims| dims.width).collect();
    widths.sort_unstable();
    widths
        .into_iter()
        .take(num_cols)
        .fold(0_u16, u16::saturating_add)
}

fn best_column_assignment(
    groups: &[MetricGroupView<'_>],
    group_dims: &[GroupDimensions],
    num_cols: usize,
    available_width: u16,
    available_height: u16,
    height_cap: Option<u16>,
    seed: Option<LayoutScore>,
) -> Option<(LayoutScore, Vec<CachedColumnPlan>)> {
    let mut assignments = vec![Vec::<usize>::new(); num_cols];
    let mut col_widths = vec![0_u16; num_cols];
    let mut col_heights = vec![0_u16; num_cols];
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&idx| {
        std::cmp::Reverse((group_dims[idx].width as u32) * (group_dims[idx].height as u32))
    });

    // min_widths[idx][j] = sum of the j smallest group widths among order[idx..],
    // so the per-node empty-column width check is O(1) instead of sorting.
    let mut min_widths = Vec::with_capacity(order.len() + 1);
    for idx in 0..=order.len() {
        let mut widths: Vec<u16> = order[idx..].iter().map(|&i| group_dims[i].width).collect();
        widths.sort_unstable();
        let mut prefix = Vec::with_capacity(widths.len() + 1);
        prefix.push(0);
        for w in widths {
            prefix.push(prefix.last().copied().unwrap_or(0_u16).saturating_add(w));
        }
        min_widths.push(prefix);
    }

    // min_height_suffix[idx] = smallest group height among order[idx..]
    // (u16::MAX when no groups remain).
    let mut min_height_suffix = vec![u16::MAX; order.len() + 1];
    for idx in (0..order.len()).rev() {
        min_height_suffix[idx] = group_dims[order[idx]]
            .height
            .min(min_height_suffix[idx + 1]);
    }

    let total_height: u32 = group_dims.iter().map(|dims| dims.height as u32).sum();
    let min_width: u16 = group_dims.iter().map(|dims| dims.width).min().unwrap_or(0);
    // Upper bound on the final total width: each column's width is the width of
    // one of its (disjoint) groups, so the sum never exceeds the widest `k`.
    let mut widths_desc: Vec<u16> = group_dims.iter().map(|dims| dims.width).collect();
    widths_desc.sort_unstable_by(|a, b| b.cmp(a));
    let max_total_width: u16 = widths_desc
        .into_iter()
        .take(num_cols)
        .fold(0_u16, u16::saturating_add);

    let mut search = ColumnSearch {
        order: &order,
        group_dims,
        min_widths: &min_widths,
        min_height_suffix: &min_height_suffix,
        total_height,
        min_width,
        max_total_width,
        available_width,
        available_height,
        height_cap,
        assignments: &mut assignments,
        col_widths: &mut col_widths,
        col_heights: &mut col_heights,
        total_width: 0,
        // Seed the internal best so branches that cannot beat the current
        // global best are pruned; a seeded best that is never beaten leaves
        // the assignments empty, which the caller ignores.
        best: seed.map(|score| (score, Vec::new())),
    };
    search.run(0);

    let (score, assignments) = search.best?;
    let columns = assignments
        .into_iter()
        .map(|mut group_indices| {
            group_indices.sort_by_key(|&idx| group_order(&groups[idx].group.key_prefix));
            build_cached_column_plan(group_dims, group_indices)
        })
        .collect();
    Some((score, columns))
}

struct ColumnSearch<'a, 'b> {
    order: &'a [usize],
    group_dims: &'a [GroupDimensions],
    min_widths: &'a [Vec<u16>],
    min_height_suffix: &'a [u16],
    total_height: u32,
    min_width: u16,
    max_total_width: u16,
    available_width: u16,
    available_height: u16,
    height_cap: Option<u16>,
    assignments: &'b mut [Vec<usize>],
    col_widths: &'b mut [u16],
    col_heights: &'b mut [u16],
    total_width: u16,
    best: Option<(LayoutScore, Vec<Vec<usize>>)>,
}

impl ColumnSearch<'_, '_> {
    fn run(&mut self, order_idx: usize) {
        #[cfg(test)]
        SEARCH_NODES.with(|counter| counter.set(counter.get() + 1));

        let remaining_groups = self.order.len().saturating_sub(order_idx);
        let empty_columns = self.empty_column_count();
        if remaining_groups < empty_columns
            || !self.can_fill_empty_columns(order_idx, empty_columns)
        {
            return;
        }

        if self.best_possible_score().is_some_and(|lower_bound| {
            self.best
                .as_ref()
                .is_some_and(|(best_score, _)| lower_bound >= *best_score)
        }) {
            return;
        }

        if order_idx == self.order.len() {
            self.record_complete_assignment();
            return;
        }

        let group_idx = self.order[order_idx];
        let dims = self.group_dims[group_idx];

        for col_idx in 0..self.assignments.len() {
            if self.assignments[col_idx].is_empty()
                && self.assignments[..col_idx].iter().any(Vec::is_empty)
            {
                continue;
            }

            let old_width = self.col_widths[col_idx];
            let old_height = self.col_heights[col_idx];
            let old_total_width = self.total_width;
            let new_width = old_width.max(dims.width);
            self.total_width = self
                .total_width
                .saturating_sub(old_width)
                .saturating_add(new_width);
            self.col_widths[col_idx] = new_width;
            self.col_heights[col_idx] = self.col_heights[col_idx].saturating_add(dims.height);

            // With a perfect (zero-waste) seed, columns may never exceed the
            // target height `H/k`, and a column whose remaining deficit is
            // smaller than every unassigned group height can never be filled
            // exactly, so those branches are pruned too.
            let within_height_cap = self.height_cap.is_none_or(|cap| {
                let height = self.col_heights[col_idx];
                height <= cap
                    && (height == cap || self.min_height_suffix[order_idx + 1] <= cap - height)
            });
            if self.total_width <= self.available_width && within_height_cap {
                self.assignments[col_idx].push(group_idx);
                self.run(order_idx + 1);
                self.assignments[col_idx].pop();
            }

            self.col_widths[col_idx] = old_width;
            self.col_heights[col_idx] = old_height;
            self.total_width = old_total_width;
        }
    }

    fn record_complete_assignment(&mut self) {
        if self.assignments.iter().any(Vec::is_empty) {
            return;
        }

        if self.total_width > self.available_width {
            return;
        }

        let max_height = self.col_heights.iter().copied().max().unwrap_or(0);
        let min_height = self.col_heights.iter().copied().min().unwrap_or(0);
        let used_area = self
            .col_widths
            .iter()
            .zip(self.col_heights.iter())
            .map(|(width, height)| (*width as u32) * (*height as u32))
            .sum::<u32>();
        let layout_area = (self.total_width as u32) * (max_height as u32);
        let num_cols = self.assignments.len() as u32;
        let max_possible_cols = self.order.len() as u32;
        let score = LayoutScore {
            overflow: max_height.saturating_sub(self.available_height) as u32,
            wasted_area: layout_area.saturating_sub(used_area),
            height_imbalance: max_height.saturating_sub(min_height) as u32,
            unused_width: self.available_width.saturating_sub(self.total_width) as u32,
            column_penalty: max_possible_cols.saturating_sub(num_cols),
            max_height: max_height as u32,
        };

        if self
            .best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            self.best = Some((score, self.assignments.to_vec()));
        }
    }

    fn empty_column_count(&self) -> usize {
        self.assignments
            .iter()
            .filter(|column| column.is_empty())
            .count()
    }

    fn can_fill_empty_columns(&self, order_idx: usize, empty_columns: usize) -> bool {
        self.total_width
            .saturating_add(self.minimum_width_for_empty_columns(order_idx, empty_columns))
            <= self.available_width
    }

    fn minimum_width_for_empty_columns(&self, order_idx: usize, empty_columns: usize) -> u16 {
        if empty_columns == 0 {
            return 0;
        }
        // Look up the precomputed sum of the `empty_columns` smallest widths
        // among the remaining groups (`min_widths[order_idx]`).
        self.min_widths[order_idx]
            .get(empty_columns)
            .copied()
            .unwrap_or(u16::MAX)
    }

    fn best_possible_score(&self) -> Option<LayoutScore> {
        let max_height = self.col_heights.iter().copied().max()?;
        let num_cols = self.assignments.len() as u32;
        // Every column ends at most `max_height` tall and at least every group
        // lands in exactly one column, so the final layout has
        // `Σ(M - H_i) = k·M - H_total ≥ k·max_height - H_total` of empty height
        // to distribute across columns that are each at least `min_width` wide.
        let wasted_area = (num_cols)
            .saturating_mul(max_height as u32)
            .saturating_sub(self.total_height)
            .saturating_mul(self.min_width as u32);
        // The shortest column can never exceed the average height, so the final
        // height imbalance is at least the current max height minus it.
        let height_imbalance =
            max_height.saturating_sub((self.total_height / num_cols) as u16) as u32;
        // Total width is at most the sum of the widest `k` groups, giving a
        // lower bound on the final unused width.
        let unused_width = self.available_width.saturating_sub(self.max_total_width) as u32;
        Some(LayoutScore {
            overflow: max_height.saturating_sub(self.available_height) as u32,
            wasted_area,
            height_imbalance,
            unused_width,
            column_penalty: (self.order.len() as u32).saturating_sub(self.assignments.len() as u32),
            max_height: max_height as u32,
        })
    }
}

#[derive(Clone, Copy)]
struct GroupDimensions {
    width: u16,
    height: u16,
}

fn build_cached_column_plan(
    group_dims: &[GroupDimensions],
    group_indices: Vec<usize>,
) -> CachedColumnPlan {
    let width = group_indices
        .iter()
        .map(|&idx| group_dims[idx].width)
        .max()
        .unwrap_or(0);
    let height = group_indices
        .iter()
        .map(|&idx| group_dims[idx].height)
        .sum();

    CachedColumnPlan {
        group_indices,
        width,
        height,
    }
}

fn build_column_plan_from_indices<'a>(
    groups: &[MetricGroupView<'a>],
    cached: &CachedColumnPlan,
) -> ColumnPlan<'a> {
    ColumnPlan {
        groups: cached
            .group_indices
            .iter()
            .map(|&idx| groups[idx].clone())
            .collect(),
        width: cached.width,
        height: cached.height,
    }
}

fn group_layout_key(
    groups: &[MetricGroupView<'_>],
    group_dims: &[GroupDimensions],
) -> Vec<GroupLayoutKey> {
    groups
        .iter()
        .zip(group_dims.iter())
        .map(|(view, dims)| GroupLayoutKey {
            name: view.group.name.clone(),
            key_prefix: view.group.key_prefix.clone(),
            entry_names: view
                .entries
                .iter()
                .map(|(name, _)| (*name).to_string())
                .collect(),
            width: dims.width,
            height: dims.height,
        })
        .collect()
}

fn required_group_width(view: &MetricGroupView<'_>, history: &MetricHistory) -> u16 {
    let title_width = display_width(&view.group.name) + 4;
    let value_col_width = view
        .entries
        .iter()
        .map(|(name, value)| display_width(&metric_value_display(view.group, name, *value)))
        .max()
        .unwrap_or(0);
    let name_col_width = view
        .entries
        .iter()
        .map(|(name, _)| display_width(&format!(" {name}")))
        .max()
        .unwrap_or(0);
    let sparkline_col_width = view
        .entries
        .iter()
        .map(|(name, _)| sparkline_required_width(view.group, name, history).saturating_sub(2))
        .max()
        .unwrap_or(0);
    let row_width = if view.entries.is_empty() {
        display_width(" (no data)") + 4
    } else {
        // Left/right borders + name column + name/sparkline gap + sparkline column
        // + sparkline/value gap + value column.
        name_col_width + sparkline_col_width + value_col_width + 6
    };

    title_width.max(row_width).min(u16::MAX as usize) as u16
}

fn desired_group_height(entry_count: usize) -> u16 {
    // Top/bottom border + title/padding, then one row per metric.
    (entry_count as u16).saturating_add(2).max(3)
}

fn group_order(prefix: &str) -> usize {
    group_sort_key(prefix).0
}

fn group_sort_key(prefix: &str) -> (usize, &str) {
    GROUPS
        .iter()
        .position(|group| group.key_prefix == prefix)
        .map(|idx| (idx, ""))
        .unwrap_or((GROUPS.len(), prefix))
}

fn custom_group_color(prefix: &str) -> Color {
    const COLORS: &[Color] = &[
        Color::LightBlue,
        Color::LightGreen,
        Color::LightMagenta,
        Color::LightCyan,
        Color::LightRed,
        Color::LightYellow,
        Color::Gray,
    ];
    let idx = prefix
        .as_bytes()
        .iter()
        .fold(0_usize, |acc, byte| acc.wrapping_add(*byte as usize))
        % COLORS.len();
    COLORS[idx]
}

/// Render a single metric group as bordered lines inside a scrollable column.
fn render_group_lines<'a>(
    width: u16,
    group: &MetricGroup,
    entries: &[(&str, f64)],
    history: &MetricHistory,
) -> Vec<Line<'a>> {
    if width == 0 {
        return Vec::new();
    }

    let width = width as usize;
    if width < 3 {
        return vec![Line::from(" "); desired_group_height(entries.len()) as usize];
    }

    let inner_width = width - 2;
    let title = format!(" {} ", group.name);
    let title_width = display_width(&title);
    let title_fill_width = inner_width.saturating_sub(title_width);
    let mut lines = vec![Line::from(vec![
        Span::raw("╭"),
        Span::styled(
            title,
            Style::default()
                .fg(group.color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("─".repeat(title_fill_width)),
        Span::raw("╮"),
    ])];

    let value_col_width = entries
        .iter()
        .map(|(name, value)| display_width(&metric_value_display(group, name, *value)))
        .max()
        .unwrap_or(0);
    let name_col_width = entries
        .iter()
        .map(|(name, _)| display_width(&format!(" {name}")))
        .max()
        .unwrap_or(0);
    let max_sparkline_available_width = inner_width
        .saturating_sub(name_col_width + value_col_width + 4)
        .min(max_sparkline_width());
    let sparkline_col_width = entries
        .iter()
        .filter_map(|(name, _)| {
            let full_key = format!("{}/{name}", group.key_prefix);
            history
                .get(&full_key)
                .and_then(|values| {
                    sparkline_for_values(group, name, values, max_sparkline_available_width)
                })
                .map(|sparkline| display_width(&sparkline))
        })
        .max()
        .unwrap_or(0);

    for &(name, value) in entries {
        let value = metric_value_display(group, name, value);
        let name_text = format!(" {name}");
        let name_width = display_width(&name_text);
        let value_width = display_width(&value);
        let full_key = format!("{}/{name}", group.key_prefix);
        let sparkline = history.get(&full_key).and_then(|values| {
            sparkline_for_values(group, name, values, max_sparkline_available_width)
        });
        let sparkline_width = sparkline.as_deref().map(display_width).unwrap_or(0);
        let name_padding = name_col_width.saturating_sub(name_width);
        let value_padding = value_col_width.saturating_sub(value_width);
        let flexible_gap = inner_width.saturating_sub(
            name_width + name_padding + sparkline_col_width + 2 + value_padding + value_width,
        );
        let sparkline_left_padding = sparkline_col_width.saturating_sub(sparkline_width);

        let mut spans = vec![
            Span::raw("│"),
            Span::styled(name_text, Style::default().fg(Color::Gray)),
            Span::raw(" ".repeat(name_padding + flexible_gap + sparkline_left_padding)),
        ];

        if let Some(sparkline) = sparkline {
            spans.push(Span::styled(sparkline, Style::default().fg(group.color)));
        }

        spans.push(Span::raw(" ".repeat(2 + value_padding)));
        spans.push(Span::styled(
            value,
            Style::default()
                .fg(group.color)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::raw("│"));
        lines.push(Line::from(spans));
    }

    if entries.is_empty() {
        let placeholder = " (no data)";
        lines.push(Line::from(vec![
            Span::raw("│"),
            Span::styled(
                placeholder,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
            Span::raw(" ".repeat(inner_width.saturating_sub(display_width(placeholder)))),
            Span::raw("│"),
        ]));
    }

    lines.push(Line::from(vec![
        Span::raw("╰"),
        Span::raw("─".repeat(inner_width)),
        Span::raw("╯"),
    ]));

    lines
}

fn sparkline_required_width(group: &MetricGroup, name: &str, history: &MetricHistory) -> usize {
    let full_key = format!("{}/{name}", group.key_prefix);
    if history.contains_key(&full_key) {
        max_sparkline_width() + 2
    } else {
        0
    }
}

fn sparkline_for_values(
    group: &MetricGroup,
    name: &str,
    values: &VecDeque<f64>,
    max_width: usize,
) -> Option<String> {
    if values.len() < 2 || max_width < 2 {
        return None;
    }

    let samples: Vec<f64> = downsample_values(values, max_width)
        .into_iter()
        .map(|value| metric_value_for_sparkline(group, name, value))
        .collect();
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    Some(
        samples
            .into_iter()
            .map(|value| {
                if range <= f64::EPSILON {
                    SPARKLINE_CHARS[0]
                } else {
                    let idx = (((value - min) / range) * (SPARKLINE_CHARS.len() - 1) as f64)
                        .round()
                        .clamp(0.0, (SPARKLINE_CHARS.len() - 1) as f64)
                        as usize;
                    SPARKLINE_CHARS[idx]
                }
            })
            .collect(),
    )
}

fn downsample_values(values: &VecDeque<f64>, max_width: usize) -> Vec<f64> {
    if values.len() <= max_width {
        return values.iter().copied().collect();
    }

    let start = values.len() - max_width;
    values.iter().skip(start).copied().collect()
}

fn max_sparkline_width() -> usize {
    SPARKLINE_HISTORY_LEN
}

/// Render the status bar at the bottom (notifications + key bindings).
fn render_empty_state(frame: &mut ratatui::Frame, area: Rect) {
    let text = Line::from(Span::styled(
        " Waiting for metrics… ",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    ));
    frame.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }), area);
}

pub(crate) fn render_status_bar(
    frame: &mut ratatui::Frame,
    area: Rect,
    notification: Option<&str>,
    scroll_offset: u16,
    max_scroll: u16,
    show_sparklines: bool,
) {
    if let Some(notification) = notification.filter(|notification| !notification.is_empty()) {
        let text = Line::from(Span::styled(
            format!(" {notification}"),
            Style::default().fg(Color::Yellow),
        ));
        frame.render_widget(Paragraph::new(text), area);
    } else {
        let scroll_hint = if max_scroll == 0 {
            String::new()
        } else {
            format!(" │ wheel/↑/↓ PgUp/PgDn scroll {scroll_offset}/{max_scroll}")
        };
        let sparkline_state = if show_sparklines { "on" } else { "off" };
        let text = Line::from(Span::styled(
            format!(
                " Q:quit S:save R:toggle-render D:toggle-deterministic P:sparklines-{sparkline_state}{scroll_hint} "
            ),
            Style::default().fg(Color::DarkGray),
        ));
        frame.render_widget(Paragraph::new(text), area);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_group_entries() {
        let mut m = HashMap::new();
        m.insert("Collect/avg step reward".into(), 1.23);
        m.insert("Collect/timesteps".into(), 1000.0);
        m.insert("GAE/time".into(), 0.5);

        let entries = group_entries(&m, "Collect");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "avg step reward");
        assert_eq!(entries[0].1, 1.23);
    }

    #[test]
    fn test_group_entries_only_includes_direct_children() {
        let mut metrics = HashMap::new();
        metrics.insert("Rewards/NormFactor/Vel2Ball".into(), 1.0);
        metrics.insert("Rewards/Total".into(), 2.0);

        let rewards = group_entries(&metrics, "Rewards");
        assert_eq!(rewards, vec![("Total", 2.0)]);

        let normalized_rewards = group_entries(&metrics, "Rewards/NormFactor");
        assert_eq!(normalized_rewards, vec![("Vel2Ball", 1.0)]);
    }

    #[test]
    fn test_sparkline_required_width_is_stable_while_history_grows() {
        let group = MetricGroup {
            name: "Loss".into(),
            key_prefix: "Loss".into(),
            color: Color::Red,
        };
        let mut history = MetricHistory::new();
        history.insert("Loss/policy".into(), VecDeque::from([1.0]));

        assert_eq!(
            sparkline_required_width(&group, "policy", &history),
            SPARKLINE_HISTORY_LEN + 2
        );

        history.insert(
            "Loss/policy".into(),
            (0..SPARKLINE_HISTORY_LEN)
                .map(|value| value as f64)
                .collect(),
        );
        assert_eq!(
            sparkline_required_width(&group, "policy", &history),
            SPARKLINE_HISTORY_LEN + 2
        );
    }

    fn loss_group() -> MetricGroup {
        MetricGroup {
            name: "Loss".into(),
            key_prefix: "Loss".into(),
            color: Color::Red,
        }
    }

    #[test]
    fn test_sparkline_for_constant_values() {
        let values = VecDeque::from([3.0, 3.0, 3.0]);
        assert_eq!(
            sparkline_for_values(&loss_group(), "policy", &values, 8),
            Some("▁▁▁".to_string())
        );
    }

    #[test]
    fn test_sparkline_is_flat_when_displayed_values_are_equal() {
        let values = VecDeque::from([1.231, 1.234]);
        assert_eq!(
            sparkline_for_values(&loss_group(), "policy", &values, 8),
            Some("▁▁".to_string())
        );
    }

    #[test]
    fn test_sparkline_respects_max_width() {
        let values = VecDeque::from([1.0, 2.0, 3.0, 4.0, 5.0]);
        let sparkline =
            sparkline_for_values(&loss_group(), "policy", &values, 3).expect("sparkline");
        assert_eq!(display_width(&sparkline), 3);
    }

    #[test]
    fn test_sparkline_too_narrow_or_short() {
        assert_eq!(
            sparkline_for_values(&loss_group(), "policy", &VecDeque::from([1.0, 2.0]), 1),
            None
        );
        assert_eq!(
            sparkline_for_values(&loss_group(), "policy", &VecDeque::from([1.0]), 8),
            None
        );
    }

    #[test]
    fn test_render_group_lines_handles_narrow_widths() {
        let group = MetricGroup {
            name: "Loss".into(),
            key_prefix: "Loss".into(),
            color: Color::Red,
        };
        let entries = [("policy", 1.0)];
        let lines = render_group_lines(2, &group, &entries, &HashMap::new());
        assert_eq!(lines.len(), desired_group_height(entries.len()) as usize);
    }

    #[test]
    fn test_can_partition_heights() {
        // 6 + 4 + 4 = 14 and 5 + 3 + 3 + 3 = 14.
        let heights = [6, 4, 4, 5, 3, 3, 3];
        assert!(can_partition_heights(&heights, 2));
        // 14 does not divide by 3.
        assert!(!can_partition_heights(&heights, 3));
        // 14 = 14 (single column always packs).
        assert!(can_partition_heights(&heights, 1));
        // More columns than groups, and zero groups.
        assert!(!can_partition_heights(&heights, 8));
        assert!(!can_partition_heights(&[], 1));
    }

    #[test]
    fn test_can_partition_heights_equal_elements() {
        // Five 3s partition into five groups of 3.
        let heights = [3, 3, 3, 3, 3];
        assert!(can_partition_heights(&heights, 5));
        // ...but not into two equal sums (15 % 2 != 0).
        assert!(!can_partition_heights(&heights, 2));
    }

    #[test]
    fn test_can_partition_heights_oversized_element() {
        // A single height larger than the target makes equal partition impossible.
        let heights = [7, 3, 2];
        assert!(!can_partition_heights(&heights, 2)); // target 6, 7 > 6
        assert!(can_partition_heights(&heights, 1));
    }
}

/// Benchmark harness for the layout planner (run with
/// `cargo test -p rlgymppo-tui --release -- --ignored --nocapture layout_bench`).
#[cfg(test)]
mod layout_bench {
    use std::time::Instant;

    use super::*;

    /// (group prefix, entry names) — mirrors a real training dashboard.
    const GROUP_DEFS: &[(&str, &[&str])] = &[
        (
            "Collect",
            &["avg step reward", "episode length", "timesteps"],
        ),
        ("GAE", &["time"]),
        ("Loss", &["policy", "value", "entropy"]),
        ("Update", &["gradient norm", "weight norm"]),
        ("Timing", &["total", "collect", "gae", "update"]),
        ("Throughput", &["steps/s"]),
        ("Cumulative", &["steps", "updates"]),
        ("Rating", &["overall"]),
        ("Rewards", &["Total", "Positive", "Negative"]),
        ("Rewards/NormFactor", &["Vel2Ball", "Vel2BallVel"]),
        ("Env", &["boost avg", "car height", "action rate"]),
        ("Skill", &["overall", "kickoff"]),
        ("Curriculum", &["level", "progress"]),
        ("Match", &["win rate", "goal diff", "avg goals"]),
        ("Rewards/Positive", &["BallTouch", "Goal"]),
        ("Rewards/Negative", &["Concede", "OwnGoal"]),
        ("Demo", &["demos", "demoed"]),
        ("Save", &["shots", "saves"]),
    ];

    fn build_groups(group_count: usize) -> Vec<MetricGroup> {
        GROUP_DEFS
            .iter()
            .take(group_count)
            .map(|(prefix, _)| MetricGroup {
                name: (*prefix).to_string(),
                key_prefix: (*prefix).to_string(),
                color: Color::Cyan,
            })
            .collect()
    }

    fn build_views(groups: &[MetricGroup]) -> Vec<MetricGroupView<'_>> {
        groups
            .iter()
            .enumerate()
            .map(|(i, group)| {
                let def = GROUP_DEFS[i].1;
                MetricGroupView {
                    group,
                    entries: def
                        .iter()
                        .enumerate()
                        .map(|(j, name)| (*name, 12_345.678 * (i as f64 + 1.0) + j as f64 * 0.001))
                        .collect(),
                }
            })
            .collect()
    }

    fn build_history(groups: &[MetricGroup]) -> MetricHistory {
        let mut history = MetricHistory::new();
        for (i, group) in groups.iter().enumerate() {
            for (j, name) in GROUP_DEFS[i].1.iter().enumerate() {
                let key = format!("{}/{name}", group.key_prefix);
                history.insert(
                    key,
                    (0..SPARKLINE_HISTORY_LEN)
                        .map(|t| t as f64 * 0.1 + i as f64 + j as f64)
                        .collect(),
                );
            }
        }
        history
    }

    /// Canonical, order-stable dump of a column plan so outputs can be diffed
    /// across implementations.
    fn dump_layout(columns: &[ColumnPlan<'_>]) -> String {
        columns
            .iter()
            .map(|column| {
                let indices: Vec<String> = column
                    .groups
                    .iter()
                    .map(|view| view.group.key_prefix.clone())
                    .collect();
                format!(
                    "{{{}}}:{}x{}",
                    indices.join(","),
                    column.width,
                    column.height
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    #[ignore]
    fn bench_layout_search() {
        let sizes: &[(u16, u16)] = &[(80, 24), (120, 30), (200, 60), (300, 100), (500, 200)];
        for &group_count in &[8usize, 10, 12, 14, 16, 18] {
            for &(w, h) in sizes {
                let groups = build_groups(group_count);
                let views = build_views(&groups);
                let history = build_history(&groups);
                let mut cache = LayoutPlanCache::default();

                reset_search_nodes();
                let start = Instant::now();
                let columns = cache.plan_metric_columns(w, h, &views, &history);
                let elapsed = start.elapsed();

                println!(
                    "groups={group_count:2} size={w:3}x{h:3} nodes={:>12} elapsed={elapsed:?} layout={}",
                    search_nodes(),
                    dump_layout(&columns),
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_partition_dp() {
        // Worst realistic heights for the partition DP: 18 groups.
        let heights = [5, 3, 5, 4, 6, 3, 4, 3, 5, 4, 5, 4, 4, 5, 4, 4, 4, 4];
        for &k in &[2usize, 4, 5, 7, 9] {
            let start = Instant::now();
            let result = can_partition_heights(&heights, k);
            println!("k={k} result={result} elapsed={:?}", start.elapsed());
        }
        // Warm cost: the DP runs on every cache miss, so measure it in a loop.
        for &k in &[2usize, 4] {
            let start = Instant::now();
            let mut result = false;
            for _ in 0..100 {
                result = can_partition_heights(&heights, k);
            }
            println!("k={k} result={result} avg warm={:?}", start.elapsed() / 100);
        }
        // Degenerate case: many equal heights forces deep submask enumeration.
        let equal = [3_u16; 18];
        let start = Instant::now();
        let result = can_partition_heights(&equal, 6);
        println!(
            "equal heights k=6 result={result} elapsed={:?}",
            start.elapsed()
        );
    }
}

/// Owned-data shim over the private layout planner, compiled only for the
/// `bench` feature so Criterion benchmarks (`benches/layout.rs`) can drive it
/// without exposing internals in the normal public API.
#[cfg(feature = "bench")]
pub mod bench {
    use std::collections::{HashMap, VecDeque};

    /// Plan metric columns for `groups` (`(prefix, [(entry, value)])`) at the
    /// given size, with sparkline history, using a fresh cache per call so the
    /// benchmark always measures the uncached (cold) planning path.
    ///
    /// Returns per-column `(group prefixes, width, height)`.
    pub fn plan_metric_columns(
        width: u16,
        height: u16,
        groups: &[(String, Vec<(String, f64)>)],
        history: &HashMap<String, VecDeque<f64>>,
    ) -> Vec<(Vec<String>, u16, u16)> {
        use super::{Color, LayoutPlanCache, MetricGroup, MetricGroupView};

        let owned: Vec<MetricGroup> = groups
            .iter()
            .map(|(name, _)| MetricGroup {
                name: name.clone(),
                key_prefix: name.clone(),
                color: Color::Cyan,
            })
            .collect();
        let views: Vec<MetricGroupView<'_>> = owned
            .iter()
            .zip(groups)
            .map(|(group, (_, entries))| MetricGroupView {
                group,
                entries: entries
                    .iter()
                    .map(|(name, value)| (name.as_str(), *value))
                    .collect(),
            })
            .collect();
        let mut cache = LayoutPlanCache::default();
        let columns = cache.plan_metric_columns(width, height, &views, history);
        columns
            .iter()
            .map(|column| {
                (
                    column
                        .groups
                        .iter()
                        .map(|view| view.group.key_prefix.clone())
                        .collect(),
                    column.width,
                    column.height,
                )
            })
            .collect()
    }
}
