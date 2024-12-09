## Steps

- Databricks CLI and configure a DEFAULT profile.
- Clone this repo.
- Open VSCode.
- Setup python .venv environment (Python 3.11, DBR ML 15.4)

## Important Links

1. Install the Databricks CLI from https://docs.databricks.com/dev-tools/cli/databricks-cli.html

2. https://docs.databricks.com/en/dev-tools/bundles/jobs-tutorial.html#create-a-bundle-using-a-project-template

3. https://docs.databricks.com/en/dev-tools/bundles/resources.html

4. https://docs.databricks.com/en/dev-tools/bundles/settings.html#bundle-mappings

5. Authenticate to your Databricks workspace, if you have not done so already:
    ```
    $ databricks configure
    ```

6. To deploy a development copy of this project, type:
    ```
    $ databricks bundle deploy --target dev
    ```
    (Note that "dev" is the default target, so the `--target` parameter
    is optional here.)

    This deploys everything that's defined for this project.
    For example, the default template would deploy a job called
    `[dev yourname] settlement_predictor_job` to your workspace.
    You can find that job by opening your workpace and clicking on **Workflows**.

7. To run a job or pipeline, use the "run" command:
   ```
   $ databricks bundle run
   ```

8. Optionally, install developer tools such as the Databricks extension for Visual Studio Code from
   https://docs.databricks.com/dev-tools/vscode-ext.html.

9. For documentation on the Databricks asset bundles format used
   for this project, and for CI/CD configuration, see
   https://docs.databricks.com/dev-tools/bundles/index.html.