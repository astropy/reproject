Making a New Reproject Release
==============================

A new release of reproject is almost fully automated.
As a mantainer it should be nice and simple to do, especially if all merged PRs
have nice titles and are correctly labelled.

Here is the process to follow to make a new release:

* Go through all the PRs since the last release, make sure they have
  descriptive titles (these will become the changelog entry) and are labelled
  correctly.
* Go to the GitHub releases interface and draft a new release, new tags should
  include the trailing `.0` on major releases. (Releases prior to 0.10.0
  didn't.)
* Use the GitHub autochange log generator, this should use the configuration in
  `.github/release.yml` to make headings based on labels.
* Edit the draft release notes as required, particularly to call out major
  changes at the top.
* Publish the release.
* Have a beverage of your choosing. (Note the wheels take a very long time to
  build).
