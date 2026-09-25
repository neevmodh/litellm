use std::time::Duration;

use litellm_http::transport::Error as TransportError;
use litellm_llms::base_llm::{
    anthropic_messages::transformation::BaseAnthropicMessagesConfig, auth::Authenticated,
};
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use serde_json::Value;

use super::{Error, common_utils::truncate_error_body};
use crate::{constants::MESSAGES_TIMEOUT_SECS, outbound::outbound_request};

pub(super) fn network(error: reqwest::Error) -> Error {
    Error::Transport(TransportError::Network(error.to_string()))
}

pub(super) async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, Error> {
    let request = outbound_request(
        authenticated,
        url.to_string(),
        body,
        Some(timeout.unwrap_or(Duration::from_secs(MESSAGES_TIMEOUT_SECS))),
    )?;
    request.send(http).await.map_err(network)
}

pub(super) async fn provider_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    match response.text().await {
        Ok(text) => Error::Transport(TransportError::Http {
            status,
            body: truncate_error_body(&text),
        }),
        Err(error) => network(error),
    }
}

pub(super) fn decode_response(
    config: &dyn BaseAnthropicMessagesConfig,
    model: &str,
    text: &str,
) -> Result<AnthropicMessagesResponse, Error> {
    let response = serde_json::from_str(text)
        .map_err(|err| Error::InvalidResponse(format!("invalid messages response JSON: {err}")))?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
}

pub(super) async fn synthesize(
    host: &super::route::MessagesHost,
    message: AnthropicMessagesResponse,
) -> Result<super::route::MessagesOutput, Error> {
    use super::route::{MessagesOutput, MessagesStreamHead};
    use litellm_host::host::Demand;

    let head = MessagesStreamHead {
        headers: vec![("content-type".into(), "text/event-stream".into())],
    };
    if host.open(head).await? == Demand::Detached {
        return Ok(MessagesOutput::Streamed);
    }
    for event in message_events(&message) {
        let bytes = bytes::Bytes::from(format!(
            "event: {}\ndata: {event}\n\n",
            event["type"].as_str().expect("generated event type")
        ));
        if host.deliver(bytes).await? == Demand::Detached {
            return Ok(MessagesOutput::Streamed);
        }
    }
    Ok(MessagesOutput::Streamed)
}

fn message_events(message: &AnthropicMessagesResponse) -> impl Iterator<Item = Value> + '_ {
    use serde_json::json;

    let initial_usage: serde_json::Map<String, Value> = [("input_tokens".into(), json!(0))]
        .into_iter()
        .chain(
            message
                .usage
                .as_ref()
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
                .map(|(key, value)| (key.clone(), value.clone())),
        )
        .chain([("output_tokens".into(), json!(0))])
        .collect();
    let start = json!({"type": "message_start", "message": AnthropicMessagesResponse {
        content: Vec::new(), stop_reason: None, stop_sequence: None,
        usage: Some(Value::Object(initial_usage)), ..message.clone()
    }});
    std::iter::once(start)
        .chain(
            message
                .content
                .iter()
                .enumerate()
                .flat_map(|(index, block)| block_events(index, block)),
        )
        .chain([
            json!({"type": "message_delta", "delta": {
                "stop_reason": message.stop_reason, "stop_sequence": message.stop_sequence,
            }, "usage": message.usage.as_ref().unwrap_or(&json!({"output_tokens": 0}))}),
            json!({"type": "message_stop"}),
        ])
}

fn block_events(index: usize, block: &Value) -> impl Iterator<Item = Value> {
    use serde_json::json;

    let (field, empty, delta) = match block["type"].as_str() {
        Some("text") => (
            "text",
            json!(""),
            Some(json!({"type": "text_delta", "text": block["text"]})),
        ),
        Some("thinking") => (
            "thinking",
            json!(""),
            Some(json!({"type": "thinking_delta", "thinking": block["thinking"]})),
        ),
        Some("tool_use" | "server_tool_use") => (
            "input",
            json!({}),
            Some(json!({"type": "input_json_delta", "partial_json": block["input"].to_string()})),
        ),
        _ => ("", Value::Null, None),
    };
    let content: serde_json::Map<String, Value> = block
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| {
            key.as_str() != field && !(field == "thinking" && key.as_str() == "signature")
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .chain((!field.is_empty()).then_some((field.into(), empty)))
        .collect();
    let signature = (field == "thinking")
        .then(|| block.get("signature"))
        .flatten()
        .map(|signature| json!({"type": "signature_delta", "signature": signature}));
    std::iter::once(
        json!({"type": "content_block_start", "index": index, "content_block": content}),
    )
    .chain(
        delta.into_iter().chain(signature).map(
            move |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
        ),
    )
    .chain([json!({"type": "content_block_stop", "index": index})])
}

pub(super) fn recover_thinking(
    error: &Error,
    provider: &str,
    body: &Value,
) -> Result<Option<serde_json::Map<String, Value>>, Error> {
    use litellm_llms::anthropic::common_utils::{
        is_anthropic_invalid_thinking_block_error, strip_thinking_blocks_from_anthropic_messages,
    };
    let Error::Transport(TransportError::Http {
        status: 400,
        body: error_text,
    }) = error
    else {
        return Ok(None);
    };
    if provider != crate::constants::ANTHROPIC_MESSAGES_PROVIDER
        || !is_anthropic_invalid_thinking_block_error(error_text)
    {
        return Ok(None);
    }
    let messages = serde_json::from_value(body["messages"].clone())
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let stripped = serde_json::to_value(strip_thinking_blocks_from_anthropic_messages(messages))
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    Ok(Some(
        body.as_object()
            .into_iter()
            .flatten()
            .filter(|(name, _)| !matches!(name.as_str(), "messages" | "thinking"))
            .map(|(name, value)| (name.clone(), value.clone()))
            .chain([("messages".into(), stripped)])
            .collect(),
    ))
}
