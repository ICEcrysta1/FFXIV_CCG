// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 保留标量书写样式的 YAML 读取；类型提升遵循共享配置已有的规则。

use std::collections::BTreeSet;
use std::ffi::CStr;
use std::fs;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::path::Path;
use std::slice;

use serde_yaml_ng::{Mapping, Value};
use unsafe_libyaml as unsafe_yaml;

pub(super) fn load_mapping(path: &Path) -> Result<Mapping, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("YAML config read failed: {}: {error}", path.display()))?;
    let parse = || {
        let mut parser = Parser::new(&text)?;
        let first = parser.next()?;
        // 整个输入都须语法正确，但只装配第一份文档。
        while parser.next()?.is_some() {}
        match first {
            None => Ok(Mapping::new()),
            Some(document) => match document.value(1, &mut BTreeSet::new())? {
                Value::Mapping(mapping) => Ok(mapping),
                _ => Err("YAML config must be a mapping".to_owned()),
            },
        }
    };
    parse().map_err(|error| format!("YAML config parse failed: {}: {error}", path.display()))
}

struct Parser<'input> {
    inner: Box<unsafe_yaml::yaml_parser_t>,
    input: PhantomData<&'input str>,
}

impl<'input> Parser<'input> {
    fn new(text: &'input str) -> Result<Self, String> {
        let mut parser = Box::<unsafe_yaml::yaml_parser_t>::new_uninit();
        // 安全：Box 提供已分配且对齐的空间；initialize 完整初始化解析器。
        if unsafe { unsafe_yaml::yaml_parser_initialize(parser.as_mut_ptr()) }.fail {
            return Err("cannot initialize YAML parser".to_owned());
        }
        // 安全：上一调用成功初始化了全部字段；Box 的堆地址保持不变。
        let mut inner = unsafe { parser.assume_init() };
        // 安全：输入由生命周期约束保持有效；解析器固定在 Box 中，内部回指不会悬空。
        unsafe {
            unsafe_yaml::yaml_parser_set_input_string(
                &mut *inner,
                text.as_ptr(),
                text.len() as u64,
            );
        }
        Ok(Self {
            inner,
            input: PhantomData,
        })
    }

    fn next(&mut self) -> Result<Option<Document>, String> {
        let mut document = MaybeUninit::uninit();
        // 安全：解析器已初始化且独占，document 指向足够大小的未初始化输出空间。
        if unsafe { unsafe_yaml::yaml_parser_load(&mut *self.inner, document.as_mut_ptr()) }.fail {
            let problem = if self.inner.problem.is_null() {
                "invalid YAML".to_owned()
            } else {
                // 安全：失败的解析器提供有效的 NUL 结尾错误文本，下一次调用前立即复制。
                unsafe { CStr::from_ptr(self.inner.problem) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(format!(
                "{problem} at line {} column {}",
                self.inner.problem_mark.line + 1,
                self.inner.problem_mark.column + 1
            ));
        }
        // 安全：load 成功后文档完整初始化；失败时库已清理，不会进入此处。
        let mut document = Document(unsafe { document.assume_init() });
        // 安全：完整文档独占可访问；库仅检查节点表，返回的指针不超出本次检查。
        if unsafe { unsafe_yaml::yaml_document_get_root_node(&mut document.0) }.is_null() {
            Ok(None)
        } else {
            Ok(Some(document))
        }
    }
}

impl Drop for Parser<'_> {
    fn drop(&mut self) {
        // 安全：inner 只在初始化成功后构造，且在此唯一释放其内部资源。
        unsafe { unsafe_yaml::yaml_parser_delete(&mut *self.inner) };
    }
}

struct Document(unsafe_yaml::yaml_document_t);

impl Document {
    fn value(&self, id: i32, active: &mut BTreeSet<i32>) -> Result<Value, String> {
        if !active.insert(id) {
            return Err("cyclic YAML alias".to_owned());
        }
        // 安全：文档在整个遍历中保持有效；查询函数只读节点表，节点编号由 composer 产生。
        let pointer = unsafe {
            unsafe_yaml::yaml_document_get_node(std::ptr::from_ref(&self.0).cast_mut(), id)
        };
        // 安全：get_node 返回文档内部的有效节点或空；借用不超过 Document 的生命周期。
        let node = unsafe { pointer.as_ref() }.ok_or_else(|| "invalid YAML node".to_owned())?;
        let value = match node.type_ {
            unsafe_yaml::YAML_SCALAR_NODE => {
                // 安全：节点类型已确认是 scalar，因此读取对应联合体字段。
                let scalar = unsafe { node.data.scalar };
                // 安全：composer 保证 scalar.value 指向文档持有的 length 字节 UTF-8 数据。
                let bytes = unsafe { slice::from_raw_parts(scalar.value, scalar.length as usize) };
                let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
                parse_scalar(text, scalar.style == unsafe_yaml::YAML_PLAIN_SCALAR_STYLE)?
            }
            unsafe_yaml::YAML_SEQUENCE_NODE => {
                // 安全：节点类型确认了联合体分支；items 的有效区间由 composer 维护。
                let items = unsafe { node.data.sequence.items };
                // 安全：start/top 位于同一分配中，区间中的每个元素都已初始化。
                let items = unsafe {
                    slice::from_raw_parts(items.start, items.top.offset_from(items.start) as usize)
                };
                Value::Sequence(
                    items
                        .iter()
                        .map(|id| self.value(*id, active))
                        .collect::<Result<_, _>>()?,
                )
            }
            unsafe_yaml::YAML_MAPPING_NODE => {
                // 安全：节点类型确认了联合体分支；pairs 的有效区间由 composer 维护。
                let pairs = unsafe { node.data.mapping.pairs };
                // 安全：start/top 位于同一分配中，区间中的每个键值节点编号都已初始化。
                let pairs = unsafe {
                    slice::from_raw_parts(pairs.start, pairs.top.offset_from(pairs.start) as usize)
                };
                let mut mapping = Mapping::new();
                for pair in pairs {
                    // 键按原始文本处理，避免把 true、42 等合法配置键改名。
                    let key = self.key(pair.key, active)?;
                    let value = self.value(pair.value, active)?;
                    if mapping.insert(Value::String(key), value).is_some() {
                        return Err("duplicate YAML mapping key".to_owned());
                    }
                }
                Value::Mapping(mapping)
            }
            _ => return Err("unsupported YAML node".to_owned()),
        };
        active.remove(&id);
        Ok(value)
    }

    fn key(&self, id: i32, active: &mut BTreeSet<i32>) -> Result<String, String> {
        // 安全：文档保持有效且查询只读；节点编号由 composer 产生。
        let pointer = unsafe {
            unsafe_yaml::yaml_document_get_node(std::ptr::from_ref(&self.0).cast_mut(), id)
        };
        // 安全：非空返回值是由 Document 持有的节点。
        let node = unsafe { pointer.as_ref() }.ok_or_else(|| "invalid YAML key".to_owned())?;
        if node.type_ == unsafe_yaml::YAML_SCALAR_NODE {
            // 安全：scalar 分支与节点类型一致，数据在 Document 存活期间有效。
            let scalar = unsafe { node.data.scalar };
            // 安全：composer 给出的字节区间已初始化，长度包含全部标量内容。
            let bytes = unsafe { slice::from_raw_parts(scalar.value, scalar.length as usize) };
            return String::from_utf8(bytes.to_vec()).map_err(|error| error.to_string());
        }
        serde_yaml_ng::to_string(&self.value(id, active)?)
            .map(|text| text.trim().to_owned())
            .map_err(|error| error.to_string())
    }
}

impl Drop for Document {
    fn drop(&mut self) {
        // 安全：Document 仅包装 load 成功的文档；该所有者负责唯一一次释放。
        unsafe { unsafe_yaml::yaml_document_delete(&mut self.0) };
    }
}

fn parse_scalar(text: &str, plain: bool) -> Result<Value, String> {
    if !plain {
        return Ok(Value::String(text.to_owned()));
    }
    match text {
        "" | "~" | "null" | "Null" | "NULL" => return Ok(Value::Null),
        "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" | "on" | "On" | "ON" => {
            return Ok(Value::Bool(true));
        }
        "false" | "False" | "FALSE" | "no" | "No" | "NO" | "off" | "Off" | "OFF" => {
            return Ok(Value::Bool(false));
        }
        _ => {}
    }
    let digits = text.strip_prefix(['+', '-']).unwrap_or(text);
    if digits.len() > 1
        && digits.starts_with('0')
        && digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!("ambiguous leading-zero integer: {text}"));
    }
    if let Ok(integer) = text.parse::<i64>() {
        return Ok(Value::Number(integer.into()));
    }
    // 仅支持十进制与标准 Infinity/NaN，不把 .inf、0x、0b 或下划线数字静默改值。
    if !matches!(text.to_ascii_lowercase().as_str(), "inf" | "+inf" | "-inf")
        && let Ok(number) = text.parse::<f64>()
    {
        return Ok(Value::Number(number.into()));
    }
    Ok(Value::String(text.to_owned()))
}
