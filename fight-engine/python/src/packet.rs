// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 只编码 pickle 2 的基础值指令；字符串复用，容器独立还原。

use std::collections::HashMap;

use serde_json::Value;

pub(crate) fn encode(value: &Value) -> Result<Vec<u8>, String> {
    let mut encoder = Encoder {
        bytes: vec![0x80, 2],
        strings: HashMap::new(),
    };
    encoder.write(value)?;
    encoder.bytes.push(b'.');
    Ok(encoder.bytes)
}

struct Encoder<'a> {
    bytes: Vec<u8>,
    strings: HashMap<&'a str, u32>,
}

impl<'a> Encoder<'a> {
    fn write(&mut self, value: &'a Value) -> Result<(), String> {
        match value {
            Value::Null => self.bytes.push(b'N'),
            Value::Bool(flag) => self.bytes.push(if *flag { 0x88 } else { 0x89 }),
            Value::String(text) => self.string(text)?,
            Value::Number(number) => {
                if let Some(number) = number.as_i64() {
                    self.integer(i128::from(number));
                } else if let Some(number) = number.as_u64() {
                    self.integer(i128::from(number));
                } else {
                    self.bytes.push(b'G');
                    self.bytes.extend_from_slice(
                        &number.as_f64().ok_or("观测包含无效浮点数")?.to_be_bytes(),
                    );
                }
            }
            Value::Array(values) => {
                self.bytes.extend_from_slice(b"](");
                for value in values {
                    self.write(value)?;
                }
                self.bytes.push(b'e');
            }
            Value::Object(values) => {
                self.bytes.extend_from_slice(b"}(");
                for (key, value) in values {
                    self.string(key)?;
                    self.write(value)?;
                }
                self.bytes.push(b'u');
            }
        }
        Ok(())
    }

    fn string(&mut self, value: &'a str) -> Result<(), String> {
        if let Some(index) = self.strings.get(value) {
            self.bytes.push(b'j');
            self.bytes.extend_from_slice(&index.to_le_bytes());
            return Ok(());
        }
        let length = u32::try_from(value.len()).map_err(|_| "观测字符串过长")?;
        let index = u32::try_from(self.strings.len()).map_err(|_| "观测字符串过多")?;
        self.bytes.push(b'X');
        self.bytes.extend_from_slice(&length.to_le_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
        self.bytes.push(b'r');
        self.bytes.extend_from_slice(&index.to_le_bytes());
        self.strings.insert(value, index);
        Ok(())
    }

    fn integer(&mut self, value: i128) {
        if let Ok(number) = i32::try_from(value) {
            self.bytes.push(b'J');
            self.bytes.extend_from_slice(&number.to_le_bytes());
        } else {
            let bytes = value.to_le_bytes();
            let mut length = bytes.len();
            // LONG1 使用最短的有符号小端整数，u64 最高位为 1 时仍保留正号。
            while length > 1
                && ((bytes[length - 1] == 0 && bytes[length - 2] & 0x80 == 0)
                    || (bytes[length - 1] == 0xff && bytes[length - 2] & 0x80 != 0))
            {
                length -= 1;
            }
            self.bytes.extend_from_slice(&[0x8a, length as u8]);
            self.bytes.extend_from_slice(&bytes[..length]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::encode;
    use serde_json::json;

    #[test]
    fn pickle_numbers_keep_integer_width_and_ieee_bits() {
        let packet = encode(&json!([
            null,
            true,
            false,
            i32::MIN,
            i64::MIN,
            u64::MAX,
            -0.0,
            1.5
        ]))
        .unwrap();
        let mut expected = b"\x80\x02](N\x88\x89J\x00\x00\x00\x80\x8a\x08".to_vec();
        expected.extend_from_slice(&i64::MIN.to_le_bytes());
        expected.extend_from_slice(b"\x8a\x09\xff\xff\xff\xff\xff\xff\xff\xff\x00");
        expected.extend_from_slice(b"G\x80\x00\x00\x00\x00\x00\x00\x00");
        expected.extend_from_slice(b"G\x3f\xf8\x00\x00\x00\x00\x00\x00e.");
        assert_eq!(packet, expected);
    }

    #[test]
    fn repeated_containers_only_share_immutable_strings() {
        let packet = encode(&json!([{"中": ["中"]}, {"中": ["中"]}])).unwrap();
        assert_eq!(
            packet,
            b"\x80\x02](}(X\x03\x00\x00\x00\xe4\xb8\xadr\x00\x00\x00\x00](j\x00\x00\x00\x00eu}(j\x00\x00\x00\x00](j\x00\x00\x00\x00eue."
        );
    }
}
