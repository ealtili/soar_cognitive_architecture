
#!/usr/bin/env python3
import click
import asyncio
import json
from pathlib import Path
from modern_soar import ModernSoarSystem, SoarAgent, demonstrate_incident_response_sync

@click.group()
def main():
    '''Modern Soar Cognitive Architecture CLI'''
    pass

@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def init(config, verbose):
    '''Initialize a new Soar system'''
    click.echo('Initializing Modern Soar system...')

    system = ModernSoarSystem()
    system.initialize_cybersecurity_domain()

    if verbose:
        click.echo(f'Knowledge graph entities: {len(system.knowledge_graph.entities)}')
        click.echo(f'Knowledge graph relationships: {len(system.knowledge_graph.relationships)}')

    click.echo('System initialized successfully!')

@main.command()
@click.option('--agent-id', '-a', required=True, help='Agent ID')
@click.option('--specialization', '-s', default='general', 
              help='Agent specialization (general, perception, planning, execution, learning)')
def create_agent(agent_id, specialization):
    '''Create a new Soar agent'''
    click.echo(f'Creating agent: {agent_id} ({specialization})')

    agent = SoarAgent(agent_id, specialization)

    click.echo(f'Agent {agent_id} created successfully!')
    return agent

@main.command()
@click.option('--incident-file', '-f', type=click.Path(exists=True),
              help='JSON file containing incident data')
@click.option('--incident-type', '-t', default='malware_detection',
              help='Type of incident to simulate')
def demo(incident_file, incident_type):
    '''Run a demonstration of the Soar system'''
    click.echo('Running Modern Soar demonstration...')

    if incident_file:
        with open(incident_file, 'r') as f:
            incident_data = json.load(f)
        click.echo(f'Loaded incident from {incident_file}')
    else:
        incident_data = {
            "type": incident_type,
            "description": f"Sample {incident_type} incident",
            "network_data": "suspicious activity detected",
            "severity": "medium"
        }
        click.echo(f'Using sample {incident_type} incident')

    # Run the demonstration
    result = demonstrate_incident_response_sync()

    click.echo('\nDemonstration Results:')
    click.echo(f'Incident ID: {result["incident_id"]}')
    click.echo(f'Severity: {result["workflow_result"]["severity"]}')
    click.echo(f'Actions: {", ".join(result["workflow_result"]["response_actions"])}')

@main.command()
@click.option('--output', '-o', type=click.Path(), default='soar_config.json',
              help='Output configuration file')
def export_config(output):
    '''Export system configuration'''
    config = {
        "system": {
            "name": "Modern Soar System",
            "version": "1.0.0",
            "initialized": True
        },
        "agents": {
            "detection_agent": {"specialization": "perception"},
            "analysis_agent": {"specialization": "planning"},
            "response_agent": {"specialization": "execution"},
            "learning_agent": {"specialization": "learning"}
        },
        "workflows": {
            "cybersecurity": {
                "enabled": True,
                "steps": ["detection", "analysis", "response", "learning"]
            }
        }
    }

    with open(output, 'w') as f:
        json.dump(config, f, indent=2)

    click.echo(f'Configuration exported to {output}')

@main.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
def serve(host, port):
    '''Start the Soar API server'''
    click.echo(f'Starting Soar API server on {host}:{port}...')

    try:
        import uvicorn
        uvicorn.run("modern_soar.api:app", host=host, port=port, reload=True)
    except ImportError:
        click.echo('Error: uvicorn not installed. Install with: pip install uvicorn')
    except KeyboardInterrupt:
        click.echo('\nServer stopped')

if __name__ == '__main__':
    main()
