"""
Specification Validator - Validates execution specifications for ThinkxLife Brain
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime

from .types import (
    AgentExecutionSpec, DataSourceSpec, ProviderSpec, ToolSpec, ProcessingSpec,
    DataSourceType
)
from .providers.provider_registry import get_provider_registry
from .data_sources import get_data_source_registry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of specification validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class SpecificationValidator:
    """
    Validates execution specifications before Brain executes them
    
    Ensures:
    - Required providers are available
    - Data source configurations are valid
    - Processing specs are within acceptable limits
    - Tools are properly configured
    """
    
    def __init__(self):
        self._initialized = False
        self.available_tools: Set[str] = set()
        self.config_limits: Dict[str, Any] = {}
        self.provider_registry = get_provider_registry()
        self.data_source_registry = get_data_source_registry()
        
        logger.info("Specification Validator initialized")
    
    async def initialize(self, brain_config: Dict[str, Any] = None) -> None:
        """Initialize validator with Brain configuration"""
        if self._initialized:
            return
        
        brain_config = brain_config or {}
        
        # Initialize provider registry first
        await self.provider_registry.initialize()
        
        # Initialize data source registry
        await self.data_source_registry.initialize()
        
        # Load configuration limits
        self._load_config_limits(brain_config)
        
        self._initialized = True
        available_providers = self.provider_registry.get_available_providers()
        available_data_sources = self.data_source_registry.get_available_sources()
        logger.info(
            f"Specification Validator initialized - "
            f"Providers: {len(available_providers)}, "
            f"Data Sources: {len(available_data_sources)}"
        )
    
    def _load_config_limits(self, brain_config: Dict[str, Any]) -> None:
        """Load configuration limits from Brain config"""
        self.config_limits = {
            "max_iterations": brain_config.get("max_iterations", 10),
            "max_timeout": brain_config.get("max_timeout", 300.0),  # 5 minutes
            "max_tokens": brain_config.get("max_tokens", 4000),
            "max_data_sources": brain_config.get("max_data_sources", 10),
            "max_tools": brain_config.get("max_tools", 5),
            "max_query_limit": brain_config.get("max_query_limit", 100)
        }
    
    async def validate_execution_spec(
        self, 
        spec: AgentExecutionSpec,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate complete execution specification
        
        Args:
            spec: AgentExecutionSpec to validate
            strict: If True, warnings are treated as errors
            
        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Validate data sources
        ds_result = self._validate_data_sources(spec.data_sources)
        errors.extend(ds_result["errors"])
        warnings.extend(ds_result["warnings"])
        suggestions.extend(ds_result["suggestions"])
        
        # Validate provider
        if spec.provider:
            prov_result = self._validate_provider(spec.provider)
            errors.extend(prov_result["errors"])
            warnings.extend(prov_result["warnings"])
            suggestions.extend(prov_result["suggestions"])
        else:
            warnings.append("No provider specified, Brain will use default")
        
        # Validate tools
        if spec.tools:
            tool_result = self._validate_tools(spec.tools)
            errors.extend(tool_result["errors"])
            warnings.extend(tool_result["warnings"])
            suggestions.extend(tool_result["suggestions"])
        
        # Validate processing configuration
        proc_result = self._validate_processing(spec.processing)
        errors.extend(proc_result["errors"])
        warnings.extend(proc_result["warnings"])
        suggestions.extend(proc_result["suggestions"])
        
        # In strict mode, treat warnings as errors
        if strict and warnings:
            errors.extend(warnings)
            warnings = []
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_data_sources(self, data_sources: List[DataSourceSpec]) -> Dict[str, List[str]]:
        """Validate data source specifications"""
        errors = []
        warnings = []
        suggestions = []
        
        if not data_sources:
            suggestions.append("No data sources specified, consider adding relevant sources")
        
        if len(data_sources) > self.config_limits["max_data_sources"]:
            errors.append(
                f"Too many data sources: {len(data_sources)} "
                f"(max: {self.config_limits['max_data_sources']})"
            )
        
        for idx, ds in enumerate(data_sources):
            # Check if source type is supported using registry
            source_type_name = ds.source_type.value if hasattr(ds.source_type, 'value') else str(ds.source_type)
            if not self.data_source_registry.is_source_available(source_type_name):
                errors.append(f"Data source {idx}: Type '{ds.source_type}' is not available")
            
            # Validate configuration using registry
            if ds.config:
                is_valid, config_errors = self.data_source_registry.validate_source_config(
                    source_type_name, ds.config
                )
                if not is_valid:
                    errors.extend([f"Data source {idx}: {err}" for err in config_errors])
            
            # Validate query limit
            if ds.limit > self.config_limits["max_query_limit"]:
                warnings.append(
                    f"Data source {idx}: Query limit {ds.limit} is high "
                    f"(recommended max: {self.config_limits['max_query_limit']})"
                )
            
            # Check for external data sources
            if ds.config and ds.config.get("db_path"):
                if self.data_source_registry.supports_external_sources(source_type_name):
                    suggestions.append(f"Data source {idx}: Using external database at {ds.config['db_path']}")
                else:
                    warnings.append(f"Data source {idx}: Type '{source_type_name}' does not support external sources")
            
            # Validate that enabled sources have required fields
            if ds.enabled:
                if ds.source_type == DataSourceType.VECTOR_DB and not ds.query and not ds.config.get("db_path"):
                    warnings.append(f"Data source {idx}: VECTOR_DB enabled but no query or db_path specified")
        
        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_provider(self, provider: ProviderSpec) -> Dict[str, List[str]]:
        """Validate provider specification using provider registry"""
        errors = []
        warnings = []
        suggestions = []
        
        # Get provider info from registry
        provider_info = self.provider_registry.get_provider_info(provider.provider_type)
        
        # Check if provider exists and is available
        if not provider_info:
            errors.append(f"Provider '{provider.provider_type}' not found")
            return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
        
        if not provider_info.available:
            errors.append(
                f"Provider '{provider.provider_type}' is not available "
                f"(missing API key or dependencies)"
            )
        
        # Validate configuration using registry
        if provider_info.available:
            config_dict = {
                "model": provider.model,
                "max_tokens": provider.max_tokens,
                "temperature": provider.temperature,
                "stream": provider.stream,
                **provider.custom_params
            }
            
            is_valid, config_errors = self.provider_registry.validate_config_for_provider(
                provider.provider_type, config_dict
            )
            
            if not is_valid:
                errors.extend(config_errors)
        
        # Validate model specifically
        if provider.model:
            valid, model_error = self.provider_registry.validate_model_for_provider(
                provider.provider_type, provider.model
            )
            if not valid:
                errors.append(f"Model validation failed: {model_error}")
        else:
            warnings.append(f"No model specified for {provider.provider_type}, using provider default")
            
            # Suggest a default model
            suggested_model = self.provider_registry.get_suggested_model_for_use_case(
                provider.provider_type, "default"
            )
            if suggested_model:
                suggestions.append(f"Consider using model '{suggested_model}' for better performance")
        
        # Provider-specific suggestions
        if provider_info:
            limits = self.provider_registry.get_provider_limits(provider.provider_type)
            
            # Check if streaming is requested but not supported
            if provider.stream and not limits.get("supports_streaming", True):
                warnings.append(f"Provider '{provider.provider_type}' may not support streaming")
            
            # Check tools support
            if provider.custom_params.get("tools") and not limits.get("supports_tools", False):
                warnings.append(f"Provider '{provider.provider_type}' may not support tools")
            
            # Temperature suggestions
            if provider.temperature < 0.0 or provider.temperature > 2.0:
                warnings.append(
                    f"Temperature {provider.temperature} is unusual (typically 0.0-1.0)"
                )
            
            # Max tokens suggestions
            if provider.max_tokens > self.config_limits["max_tokens"]:
                warnings.append(
                    f"max_tokens {provider.max_tokens} exceeds recommended limit "
                    f"({self.config_limits['max_tokens']})"
                )
            
            if provider.stream and provider.max_tokens > 2000:
                suggestions.append("Consider reducing max_tokens for better streaming performance")
        
        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_tools(self, tools: List[ToolSpec]) -> Dict[str, List[str]]:
        """Validate tool specifications"""
        errors = []
        warnings = []
        suggestions = []
        
        if len(tools) > self.config_limits["max_tools"]:
            warnings.append(
                f"Many tools specified ({len(tools)}), may impact performance"
            )
        
        for idx, tool in enumerate(tools):
            # Check if tool type is valid (basic validation)
            if not tool.tool_type or not isinstance(tool.tool_type, str):
                errors.append(f"Tool {idx}: Invalid tool_type")
            
            # Check if enabled tools have config
            if tool.enabled and not tool.config:
                warnings.append(f"Tool {idx} ({tool.tool_type}): No configuration provided")
        
        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_processing(self, processing: ProcessingSpec) -> Dict[str, List[str]]:
        """Validate processing specification"""
        errors = []
        warnings = []
        suggestions = []
        
        # Validate iterations
        if processing.max_iterations > self.config_limits["max_iterations"]:
            errors.append(
                f"max_iterations {processing.max_iterations} exceeds limit "
                f"({self.config_limits['max_iterations']})"
            )
        
        if processing.max_iterations < 1:
            errors.append("max_iterations must be at least 1")
        
        # Validate timeout
        if processing.timeout_seconds > self.config_limits["max_timeout"]:
            errors.append(
                f"timeout_seconds {processing.timeout_seconds} exceeds limit "
                f"({self.config_limits['max_timeout']})"
            )
        
        if processing.timeout_seconds < 1.0:
            errors.append("timeout_seconds must be at least 1.0")
        
        # Suggestions
        if processing.max_iterations > 5:
            suggestions.append("High iteration count may increase latency")
        
        if processing.enable_conversation_memory and processing.max_iterations > 3:
            suggestions.append("Consider reducing iterations when using conversation memory")
        
        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
    
    def validate_quick(self, spec: AgentExecutionSpec) -> bool:
        """
        Quick validation - returns True if spec is executable
        
        Only checks critical errors, skips warnings and suggestions
        """
        # Check provider availability using registry
        if spec.provider:
            if not self.provider_registry.is_provider_available(spec.provider.provider_type):
                return False
            
            # Quick model validation
            if spec.provider.model:
                valid, _ = self.provider_registry.validate_model_for_provider(
                    spec.provider.provider_type, spec.provider.model
                )
                if not valid:
                    return False
        
        # Check basic limits
        if spec.processing.max_iterations > self.config_limits["max_iterations"]:
            return False
        
        if spec.processing.timeout_seconds > self.config_limits["max_timeout"]:
            return False
        
        # Check data source paths
        for ds in spec.data_sources:
            if ds.config and ds.config.get("db_path"):
                import os
                if not os.path.exists(ds.config["db_path"]):
                    return False
        
        return True
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report"""
        lines = []
        
        if result.valid:
            lines.append(" Specification is valid and executable")
        else:
            lines.append(" Specification has errors and cannot be executed")
        
        if result.errors:
            lines.append(f"\n Errors ({len(result.errors)}):")
            for error in result.errors:
                lines.append(f"  • {error}")
        
        if result.warnings:
            lines.append(f"\n Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                lines.append(f"  • {warning}")
        
        if result.suggestions:
            lines.append(f"\n Suggestions ({len(result.suggestions)}):")
            for suggestion in result.suggestions:
                lines.append(f"  • {suggestion}")
        
        return "\n".join(lines)


# Singleton instance
_spec_validator_instance: Optional[SpecificationValidator] = None


def get_spec_validator() -> SpecificationValidator:
    """Get singleton SpecificationValidator instance"""
    global _spec_validator_instance
    if _spec_validator_instance is None:
        _spec_validator_instance = SpecificationValidator()
    return _spec_validator_instance

