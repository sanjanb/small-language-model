"""Command-line interface for the document extraction system."""
import click
import json
import logging
from pathlib import Path
from typing import Optional

from .pipeline import DocumentProcessor
from .config import Config


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """Document Text Extraction System using OCR and NER."""
    # Setup logging level
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Store config path in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory for results')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', 
              help='Output format for structured data')
@click.pass_context
def process(ctx, input_file: str, output: Optional[str], format: str):
    """Process a single document file."""
    try:
        # Initialize processor
        processor = DocumentProcessor(ctx.obj.get('config'))
        
        # Update output format in config
        processor.config.output.format = format
        
        click.echo(f"Processing document: {input_file}")
        
        # Process the document
        result = processor.process_document(input_file, output)
        
        if result.success:
            click.echo(f"✓ Successfully processed in {result.processing_time:.2f}s")
            click.echo(f"  OCR confidence: {result.ocr_result.confidence:.2f}")
            click.echo(f"  NER confidence: {result.ner_result.confidence_score:.2f}")
            click.echo(f"  Entities found: {len(result.ner_result.entities)}")
            
            if not output:
                # Display results inline
                click.echo("\n--- Extracted Text ---")
                click.echo(result.ocr_result.text[:500] + "..." if len(result.ocr_result.text) > 500 else result.ocr_result.text)
                
                click.echo("\n--- Key Information ---")
                key_info = result.ner_result.structured_data.get('key_information', {})
                for key, value in key_info.items():
                    if isinstance(value, dict) and 'value' in value:
                        click.echo(f"  {key}: {value['value']} (confidence: {value['confidence']:.2f})")
                    else:
                        click.echo(f"  {key}: {value}")
        else:
            click.echo(f"✗ Processing failed: {result.error_message}")
            raise click.ClickException(result.error_message)
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--pattern', '-p', multiple=True, 
              help='File patterns to process (e.g., *.jpg, *.png)')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json',
              help='Output format for structured data')
@click.pass_context
def batch(ctx, input_dir: str, output_dir: str, pattern: tuple, format: str):
    """Process multiple documents in a directory."""
    try:
        # Initialize processor
        processor = DocumentProcessor(ctx.obj.get('config'))
        
        # Update output format in config
        processor.config.output.format = format
        
        # Use provided patterns or defaults
        patterns = list(pattern) if pattern else None
        
        click.echo(f"Processing documents from: {input_dir}")
        click.echo(f"Output directory: {output_dir}")
        
        # Process batch
        results = processor.process_batch(input_dir, output_dir, patterns)
        
        # Display summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        
        click.echo(f"\n--- Batch Processing Summary ---")
        click.echo(f"Total files: {total}")
        click.echo(f"Successful: {successful}")
        click.echo(f"Failed: {total - successful}")
        click.echo(f"Success rate: {successful/total*100:.1f}%")
        
        # Show failed files
        failed_files = [path for path, result in results.items() if not result.success]
        if failed_files:
            click.echo(f"\nFailed files:")
            for file_path in failed_files:
                click.echo(f"  ✗ {file_path}")
        
        click.echo(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument('config_path', type=click.Path())
def init_config(config_path: str):
    """Create a default configuration file."""
    try:
        config = Config()
        config.to_yaml(config_path)
        click.echo(f"Created default configuration file: {config_path}")
        click.echo("You can edit this file to customize the processing settings.")
    except Exception as e:
        click.echo(f"Error creating config file: {str(e)}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.pass_context
def info(ctx):
    """Display system information and supported formats."""
    try:
        processor = DocumentProcessor(ctx.obj.get('config'))
        
        click.echo("Document Text Extraction System")
        click.echo("=" * 40)
        
        click.echo(f"\nSupported formats: {', '.join(processor.get_supported_formats())}")
        
        click.echo(f"\nConfiguration:")
        click.echo(f"  OCR Engine: {processor.config.ocr.engine}")
        click.echo(f"  OCR Language: {processor.config.ocr.language}")
        click.echo(f"  NER Model: {processor.config.ner.model_name}")
        click.echo(f"  Output Format: {processor.config.output.format}")
        
        click.echo(f"\nCustom entity types:")
        for entity_type in processor.config.ner.custom_entities:
            click.echo(f"  - {entity_type}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.ClickException(str(e))


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.pass_context
def validate(ctx, input_file: str):
    """Validate if a file can be processed."""
    try:
        processor = DocumentProcessor(ctx.obj.get('config'))
        
        if processor.validate_input(input_file):
            click.echo(f"✓ {input_file} is valid and can be processed")
        else:
            click.echo(f"✗ {input_file} is not valid or not supported")
            raise click.ClickException(f"Unsupported file: {input_file}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.ClickException(str(e))


if __name__ == '__main__':
    cli()