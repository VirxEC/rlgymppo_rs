use std::collections::HashMap;

use unicode_width::UnicodeWidthStr;

use crate::render::MetricGroup;

pub(crate) fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

pub(crate) fn metric_value_display(
    group: &MetricGroup,
    name: &str,
    value: f64,
    prev_vals: &HashMap<String, f64>,
) -> String {
    if group.key_prefix == "Rating" {
        // Show delta alongside the current rating (1 decimal).
        let full_key = format!("Rating/{name}");
        let prev = prev_vals.get(&full_key).copied().unwrap_or(value);
        let delta = value - prev;
        let val_str = format!("{value:.1}");
        if delta.abs() >= 0.05 {
            let sign = if delta >= 0.0 { '+' } else { '-' };
            let delta_str = format!("{:.1}", delta.abs());
            format!("{val_str} ({sign}{delta_str})")
        } else {
            val_str
        }
    } else if group.key_prefix == "Throughput"
        || (group.key_prefix == "Collect" && name == "episode length")
    {
        format_num(value.round())
    } else {
        format_num(value)
    }
}

/// Format a number for display (commas for integers, 4-decimal for floats).
pub(crate) fn format_num(val: f64) -> String {
    if val == (val as i64) as f64 {
        let int_val = val as i64;
        // Add thousands separators.
        let s = int_val.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        let negative = int_val < 0;
        let abs_s = if negative { &s[1..] } else { &s };
        for (i, ch) in abs_s.chars().enumerate() {
            if i > 0 && (abs_s.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(ch);
        }
        if negative {
            result.insert(0, '-');
        }
        result
    } else if val.abs() < 1e-3 {
        format!("{val:.2e}")
    } else if val.abs() < 1.0 {
        format!("{val:.4}")
    } else {
        format!("{val:.2}")
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_format_num_integer() {
        assert_eq!(format_num(1_234_567.0), "1,234,567");
        assert_eq!(format_num(0.0), "0");
        assert_eq!(format_num(-1000.0), "-1,000");
    }

    #[test]
    fn test_format_num_float() {
        assert_eq!(format_num(PI), "3.1416");
        assert_eq!(format_num(0.00123), "0.0012");
    }

    #[test]
    fn test_format_num_special_values() {
        assert_eq!(format_num(f64::NAN), "NaN");
        assert_eq!(format_num(f64::INFINITY), "inf");
        assert_eq!(format_num(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_num(-0.0), "0");
    }
}
